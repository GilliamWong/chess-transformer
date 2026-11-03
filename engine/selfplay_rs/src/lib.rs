//mcts.py's search for many trees at once while python runs the network: self play games (SelfPlayGames) and match searches (Searches)
mod affinity;
mod concurrent;
mod fake_cuda;
mod games;
mod rounds;
mod rules;
mod search;

use games::{FinishedGame, Game, GameSettings};
use pyo3::buffer::{PyBuffer, ReadOnlyCell};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use rules::{
    MOVE_ID_TABLE_LENGTH, MoveIds, PackedMove, TOKENS_PER_BOARD, Tables, encode_board, final_result, game_result,
    legal_moves_in_python_order, position_key,
};
use search::{Cache, Round, SearchSettings, SharedCache, SharedTree, Tree};
use shakmaty::fen::Fen;
use shakmaty::uci::UciMove;
use shakmaty::{CastlingMode, Chess, Position};
use shakmaty_syzygy::SyzygyError;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;

fn tablebase_error(error: SyzygyError) -> PyErr {
    PyRuntimeError::new_err(format!("syzygy tables: {error}"))
}

fn move_ids(table: Vec<i16>) -> PyResult<MoveIds> {
    if table.len() != MOVE_ID_TABLE_LENGTH {
        return Err(PyValueError::new_err(format!("the move id table has {} entries, not {MOVE_ID_TABLE_LENGTH}", table.len())));
    }
    Ok(MoveIds::new(table))
}

fn open_tables(folder: Option<&str>, pieces: usize) -> PyResult<Option<Tables>> {
    folder.map(|folder| Tables::open(folder, pieces).map_err(|error| PyRuntimeError::new_err(format!("{folder}: {error}")))).transpose()
}

//a board from its fen and moves, with the key of every position along the way
fn play_out(fen: &str, moves: &[String]) -> PyResult<(Chess, Vec<u64>)> {
    let fen: Fen = fen.parse().map_err(|error| PyValueError::new_err(format!("{fen}: {error}")))?;
    let mut board: Chess = fen.into_position(CastlingMode::Standard).map_err(|error| PyValueError::new_err(format!("{error}")))?;
    let mut keys = vec![position_key(&board)];
    play_moves(&mut board, &mut keys, moves)?;
    Ok((board, keys))
}

fn play_moves(board: &mut Chess, keys: &mut Vec<u64>, moves: &[String]) -> PyResult<()> {
    for uci in moves {
        let illegal = |error: &dyn std::fmt::Display| PyValueError::new_err(format!("{uci}: {error}"));
        let chess_move = uci.parse::<UciMove>().map_err(|error| illegal(&error))?.to_move(board).map_err(|error| illegal(&error))?;
        board.play_unchecked(chess_move);
        keys.push(position_key(board));
    }
    Ok(())
}

fn answer_numbers<'a>(py: Python<'a>, answers: &'a PyBuffer<f32>) -> PyResult<&'a [ReadOnlyCell<f32>]> {
    answers.as_slice(py).ok_or_else(|| PyValueError::new_err("the answers must be one contiguous array"))
}

//an answer's number: a cell of the caller's array, or an f32 of the pinned answers the round loop reads
trait Number {
    fn number(&self) -> f32;
}

impl Number for ReadOnlyCell<f32> {
    fn number(&self) -> f32 {
        self.get()
    }
}

impl Number for f32 {
    fn number(&self) -> f32 {
        *self
    }
}

fn answers_for<N: Number>(answers: &[N], boards: usize, answer_width: usize) -> PyResult<&[N]> {
    if answers.len() != boards * answer_width {
        return Err(PyValueError::new_err(format!("{} numbers of answers for {boards} boards of {answer_width} each", answers.len())));
    }
    Ok(answers)
}

fn uci(chess_move: PackedMove) -> String {
    chess_move.to_uci().to_string()
}

#[pyclass(unsendable)]
struct SelfPlayGames {
    settings: GameSettings,
    move_ids: MoveIds,
    answer_width: usize,
    tables: Option<Tables>,
    seed: u64,
    start: Chess,
    start_keys: Vec<u64>,
    start_plies: usize,
    games: Vec<Game>,
    waiting: Vec<usize>, //games whose leaf the network is evaluating, in board order
    finished: Vec<FinishedGame>,
    cache: Option<Cache>,
    simulations_per_round: u32,
}

#[pymethods]
impl SelfPlayGames {
    //simulations_per_round: a game's round may do this many simulations, all but the last answered at once (by the cache or the game's end)
    #[new]
    #[pyo3(signature = (*, move_id_table, answer_width, start_fen, start_moves, seed, simulations, sampled_plies, max_plies, tree_reuse,
                        top_moves, no_move, c_puct, c_puct_factor, c_puct_base, fpu_reduction, draw, repetitions_for_a_draw,
                        root_noise_alpha, root_noise_share, tablebase_folder, tablebase_pieces, cache_positions=0, simulations_per_round=1))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        move_id_table: Vec<i16>,
        answer_width: usize,
        start_fen: &str,
        start_moves: Vec<String>,
        seed: u64,
        simulations: u32,
        sampled_plies: usize,
        max_plies: usize,
        tree_reuse: bool,
        top_moves: usize,
        no_move: i16,
        c_puct: f64,
        c_puct_factor: f64,
        c_puct_base: f64,
        fpu_reduction: f64,
        draw: f64,
        repetitions_for_a_draw: usize,
        root_noise_alpha: f64,
        root_noise_share: f64,
        tablebase_folder: Option<&str>,
        tablebase_pieces: usize,
        cache_positions: usize,
        simulations_per_round: u32,
    ) -> PyResult<SelfPlayGames> {
        if simulations_per_round == 0 {
            return Err(PyValueError::new_err("a round needs at least one simulation"));
        }
        let (start, start_keys) = play_out(start_fen, &start_moves)?;
        Ok(SelfPlayGames {
            settings: GameSettings {
                search: SearchSettings {
                    c_puct,
                    c_puct_factor,
                    c_puct_base,
                    fpu_reduction,
                    draw,
                    repetitions_for_a_draw,
                    value_sharpness: 1.0,
                    certainty: false,
                    transpositions: false,
                },
                simulations,
                sampled_plies,
                max_plies,
                tree_reuse,
                top_moves,
                no_move,
                root_noise_alpha,
                root_noise_share,
            },
            move_ids: move_ids(move_id_table)?,
            answer_width,
            tables: open_tables(tablebase_folder, tablebase_pieces)?,
            seed,
            start,
            start_keys,
            start_plies: start_moves.len(),
            games: Vec::new(),
            waiting: Vec::new(),
            finished: Vec::new(),
            cache: (cache_positions > 0).then(|| Cache::new(cache_positions)),
            simulations_per_round,
        })
    }

    //false when the game was over before it began (it is then finished, without positions)
    fn start_game(&mut self, number: u64) -> PyResult<bool> {
        let game = Game::new(number, self.seed, &self.start, &self.start_keys, self.start_plies);
        match game.final_result(self.settings.max_plies, self.tables.as_ref()).map_err(tablebase_error)? {
            Some(white_score) => {
                self.finished.push(game.into_finished(white_score));
                Ok(false)
            }
            None => {
                self.games.push(game);
                Ok(true)
            }
        }
    }

    //the boards of the leaves that need the network, at most one a game (TOKENS_PER_BOARD bytes each); until a root's noise is in,
    //its round is one simulation
    fn descend<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        if !self.waiting.is_empty() {
            return Err(PyRuntimeError::new_err("the last leaves are still waiting for finish"));
        }
        let mut boards = Vec::new();
        for (index, game) in self.games.iter_mut().enumerate() {
            let round = Round {
                id: 0, //one round out at a time
                leaves: 1,
                simulations: if game.awaits_noise() { 1 } else { self.simulations_per_round },
                collisions: 0,
                visit_limit: self.settings.simulations.max(game.tree.root().visit_count + 1),
                memory_limit: usize::MAX,
            };
            if game
                .tree
                .gather(&round, &self.settings.search, self.tables.as_ref(), self.cache.as_ref(), &mut boards)
                .map_err(tablebase_error)?
                > 0
            {
                self.waiting.push(index);
            }
        }
        Ok(PyBytes::new(py, &boards))
    }

    //finishes the waiting simulations; every game whose root has its simulations plays a move. Returns how many ended
    fn finish(&mut self, py: Python<'_>, answers: PyBuffer<f32>) -> PyResult<usize> {
        let answers = answers_for(answer_numbers(py, &answers)?, self.waiting.len(), self.answer_width)?;
        for (row, &index) in self.waiting.iter().enumerate() {
            let answer = &answers[row * self.answer_width..(row + 1) * self.answer_width];
            self.games[index].tree.finish_leaf(
                0,
                |column| answer[column].get(),
                self.answer_width,
                &self.move_ids,
                self.cache.as_mut(),
                &self.settings.search,
            );
        }
        self.waiting.clear();

        let mut ended = 0;
        let mut playing = Vec::with_capacity(self.games.len());
        for mut game in self.games.drain(..) {
            game.add_noise_once_expanded(&self.settings);
            if game.tree.root().visit_count >= self.settings.simulations {
                game.play_move(&self.settings, &self.move_ids);
                if let Some(white_score) = game.final_result(self.settings.max_plies, self.tables.as_ref()).map_err(tablebase_error)? {
                    self.finished.push(game.into_finished(white_score));
                    ended += 1;
                    continue;
                }
            }
            playing.push(game);
        }
        self.games = playing;
        Ok(ended)
    }

    //the games that ended, as (number, boards, visit_moves, visit_shares, root_scores, white_to_move, white_score) with
    //the arrays as bytes: uint8, int16, float32, float32 and bool
    #[allow(clippy::type_complexity)]
    fn take_finished<'py>(
        &mut self,
        py: Python<'py>,
    ) -> Vec<(u64, Bound<'py, PyBytes>, Bound<'py, PyBytes>, Bound<'py, PyBytes>, Bound<'py, PyBytes>, Bound<'py, PyBytes>, f64)> {
        self.finished
            .drain(..)
            .map(|game| {
                let record = game.record;
                let visit_moves: Vec<u8> = record.visit_moves.iter().flat_map(|id| id.to_le_bytes()).collect();
                let visit_shares: Vec<u8> = record.visit_shares.iter().flat_map(|share| share.to_le_bytes()).collect();
                let root_scores: Vec<u8> = record.root_scores.iter().flat_map(|score| score.to_le_bytes()).collect();
                (
                    game.number,
                    PyBytes::new(py, &record.boards),
                    PyBytes::new(py, &visit_moves),
                    PyBytes::new(py, &visit_shares),
                    PyBytes::new(py, &root_scores),
                    PyBytes::new(py, &record.white_to_move),
                    game.white_score,
                )
            })
            .collect()
    }

    fn playing(&self) -> bool {
        !self.games.is_empty() || !self.finished.is_empty()
    }

    //for the parity tests: the position after `moves`' encoding, legal move ids in order, game_result and final_result
    #[allow(clippy::type_complexity)]
    fn inspect<'py>(&self, py: Python<'py>, moves: Vec<String>) -> PyResult<(Bound<'py, PyBytes>, Vec<usize>, Option<f64>, Option<f64>)> {
        let (mut board, mut keys) = (self.start.clone(), self.start_keys.clone());
        play_moves(&mut board, &mut keys, &moves)?;
        let mut encoding = Vec::new();
        encode_board(&board, &mut encoding);
        let legal_moves = legal_moves_in_python_order(&board);
        let move_ids = legal_moves.iter().map(|&chess_move| self.move_ids.id(chess_move, board.turn())).collect();
        let tables = self.tables.as_ref();
        let search = &self.settings.search;
        let in_search =
            game_result(&board, &keys, &legal_moves, search.draw, search.repetitions_for_a_draw, tables).map_err(tablebase_error)?;
        let over = final_result(&board, &keys, self.start_plies + moves.len(), self.settings.max_plies, tables).map_err(tablebase_error)?;
        Ok((PyBytes::new(py, &encoding), move_ids, in_search, over))
    }
}

//one search of a match: mcts.search without noise
struct Search {
    id: u64,
    fen: String,
    moves: Vec<String>,
    simulations: u32,
    stopped: bool,
    tree: Tree,
}

impl Search {
    fn done(&self) -> bool {
        (self.stopped || self.tree.root().visit_count >= self.simulations || self.tree.full) && self.tree.pending() == 0
    }

    //as player.search_move reports it
    fn report(&self) -> SearchReport {
        let tree = &self.tree;
        let line = tree.principal_variation();
        //one simulation visits no child: then the root's score for its side to move
        let score = match tree.child(tree.root_index(), line[0]) {
            Some(chosen) if chosen.visit_count > 0 => chosen.mean_score(),
            _ => 1.0 - tree.root().mean_score(),
        };
        let expected_line = line.iter().map(|&edge| uci(tree.chess_move(edge))).collect();
        let root_moves = tree.edges_by_visits().into_iter().map(|edge| (uci(tree.chess_move(edge)), tree.visits(tree.root_index(), edge)));
        (self.id, expected_line, score, root_moves.collect(), tree.root().visit_count)
    }
}

//(id, expected line in uci, the chosen move's mean score, root moves by visits, the root's visits)
type SearchReport = (u64, Vec<String>, f64, Vec<(String, u32)>, u32);

//a round whose leaves the network is evaluating: its number in the trees (search.rs Round's id), the name descend gave it,
//and the searches with leaves in it with how many, in board order
struct RoundOut {
    number: u64,
    name: Option<u64>,
    leaves: Vec<(u64, u32)>,
    lean: bool,
}

//a lean round's answer row: its legal moves' logits in python-chess's order, padded to LEGAL_SLOTS (chess's most legal moves)
//with the expected score, which also ends them
const LEGAL_SLOTS: usize = 218;
const LEAN_COLUMNS: usize = LEGAL_SLOTS + 1;
//a lean board's bytes: its tokens, then its columns as u16s
const LEAN_ROW: usize = TOKENS_PER_BOARD + 2 * LEAN_COLUMNS;

//the tree of a game's last search, for its next search at the position after `moves`
struct KeptTree<T = Tree> {
    fen: String,
    moves: Vec<String>,
    tree: Kept<T>,
}

//a kept tree as its last search left it, on advance's thread, or back from it with the leaves the next root may make stale
//(None if the thread was stopped before it found them all)
enum Kept<T = Tree> {
    Searched(T),
    Advancing(JoinHandle<(T, Option<Vec<Vec<u16>>>)>, Arc<Stops>),
    Advanced(T, Option<Vec<Vec<u16>>>),
}

//what of an advance's thread is to stop at its next look: the copy, and the walk too once the tree is unwanted
#[derive(Default)]
struct Stops {
    copy: AtomicBool,
    walk: AtomicBool,
}

impl<T> Kept<T> {
    //the tree to search on: an advance not done stops its copy and is waited for (its walk is the shortest way to the stale leaves)
    fn ready(self) -> Option<Kept<T>> {
        match self {
            Kept::Advancing(thread, stops) => {
                stops.copy.store(true, Ordering::Relaxed);
                thread.join().ok().map(|(tree, candidates)| Kept::Advanced(tree, candidates))
            }
            kept => Some(kept),
        }
    }

    fn abandon(self) {
        if let Kept::Advancing(_, stops) = self {
            stops.copy.store(true, Ordering::Relaxed);
            stops.walk.store(true, Ordering::Relaxed);
        }
    }
}

//what Searches' calls change, behind its lock
struct State {
    searches: Vec<Search>,
    rounds: VecDeque<RoundOut>, //oldest first
    rounds_descended: u64,
    kept: HashMap<u64, KeptTree>,
    cache: Option<Cache>,
    //concurrent_descents' searches and kept trees, in place of searches and kept
    shared_searches: Vec<concurrent::SharedSearch>,
    shared_kept: HashMap<u64, KeptTree<Arc<SharedTree>>>,
}

//every call releases the gil while it works, and python threads may share one Searches: a call waits for the one before it to end
#[pyclass(frozen)]
struct Searches {
    settings: SearchSettings,
    move_ids: MoveIds,
    answer_width: usize,
    tables: Option<Tables>,
    tree_reuse: bool,
    memory_limit: usize, //bytes a search's tree may take, usize::MAX for no limit
    advance_threads: usize,
    //the searches are SharedTrees (search/shared.rs) that the networks' threads work in at once
    concurrent_descents: bool,
    shared_cache: Option<SharedCache>,
    state: Mutex<State>,
}

//(the root's visits, the visits of its two most visited moves)
type RootNumbers = (u32, u32, u32);

fn find_search(state: &State, id: u64) -> PyResult<&Search> {
    state.searches.iter().find(|search| search.id == id).ok_or_else(|| PyValueError::new_err(format!("no search {id}")))
}

//the chosen move's score, or the root's for its side to move before the move has a visit
fn root_score_of(search: &Search) -> f64 {
    let tree = &search.tree;
    match tree.principal_variation().first().and_then(|&edge| tree.child(tree.root_index(), edge)) {
        Some(chosen) if chosen.visit_count > 0 => chosen.mean_score(),
        _ => 1.0 - tree.root().mean_score(),
    }
}

//None when search id is not running
fn root_numbers(state: &State, id: u64) -> Option<RootNumbers> {
    let search = state.searches.iter().find(|search| search.id == id)?;
    let (most, second) = search.tree.two_most_visits();
    Some((search.tree.root().visit_count, most, second))
}

//descend_into's array of boards: one byte a token, or an int64 (the engine's lanes keep their boards as int64s)
enum BoardsBuffer {
    Bytes(PyBuffer<u8>),
    Wide(PyBuffer<i64>),
}

impl BoardsBuffer {
    fn get(boards: &Bound<'_, PyAny>) -> PyResult<BoardsBuffer> {
        let target = match PyBuffer::<i64>::get(boards) {
            Ok(wide) => BoardsBuffer::Wide(wide),
            Err(_) => BoardsBuffer::Bytes(PyBuffer::<u8>::get(boards)?),
        };
        let (readonly, contiguous) = match &target {
            BoardsBuffer::Bytes(buffer) => (buffer.readonly(), buffer.is_c_contiguous()),
            BoardsBuffer::Wide(buffer) => (buffer.readonly(), buffer.is_c_contiguous()),
        };
        if readonly || !contiguous {
            return Err(PyValueError::new_err("the boards must go into one writable contiguous array"));
        }
        Ok(target)
    }

    //the boards it has room for, at a row of `width` items each
    fn boards(&self, width: usize) -> usize {
        match self {
            BoardsBuffer::Bytes(buffer) => buffer.item_count() / width,
            BoardsBuffer::Wide(buffer) => buffer.item_count() / width,
        }
    }

    //the tokens, a board's TOKENS_PER_BOARD after another; with columns, lean rows (LEAN_ROW bytes, columns as little-endian u16s)
    fn write(&self, py: Python<'_>, tokens: &[u8], columns: Option<&[u16]>) -> PyResult<()> {
        let unwritable = || PyValueError::new_err("the boards must go into one writable contiguous array");
        match (self, columns) {
            (BoardsBuffer::Bytes(buffer), None) => {
                for (cell, &token) in buffer.as_mut_slice(py).ok_or_else(unwritable)?.iter().zip(tokens) {
                    cell.set(token);
                }
            }
            (BoardsBuffer::Bytes(buffer), Some(columns)) => {
                let cells = buffer.as_mut_slice(py).ok_or_else(unwritable)?;
                debug_assert_eq!(tokens.len() / TOKENS_PER_BOARD, columns.len() / LEAN_COLUMNS);
                let (boards, board_columns) = (tokens.as_chunks::<TOKENS_PER_BOARD>().0, columns.as_chunks::<LEAN_COLUMNS>().0);
                let rows = cells.as_chunks::<LEAN_ROW>().0;
                for (row, (board, board_columns)) in rows.iter().zip(boards.iter().zip(board_columns)) {
                    let (row_tokens, row_columns) = row.split_at(TOKENS_PER_BOARD);
                    for (cell, &token) in row_tokens.iter().zip(board) {
                        cell.set(token);
                    }
                    for (cells, column) in row_columns.as_chunks::<2>().0.iter().zip(board_columns) {
                        let [low, high] = column.to_le_bytes();
                        cells[0].set(low);
                        cells[1].set(high);
                    }
                }
            }
            (BoardsBuffer::Wide(buffer), None) => {
                for (cell, &token) in buffer.as_mut_slice(py).ok_or_else(unwritable)?.iter().zip(tokens) {
                    cell.set(i64::from(token));
                }
            }
            (BoardsBuffer::Wide(_), Some(_)) => return Err(PyValueError::new_err("lean rows go into an array of bytes")),
        }
        Ok(())
    }
}

//the network's answers read without the gil: the caller leaves their array alone until finish returns
struct Answers<'a>(&'a [ReadOnlyCell<f32>]);

//SAFETY: the cells are only read, and the array lives as long as the buffer finish holds
unsafe impl Send for Answers<'_> {}

impl<'a> Answers<'a> {
    //a method, so that a closure takes the whole of Answers (Send) rather than its field
    fn numbers(self) -> &'a [ReadOnlyCell<f32>] {
        self.0
    }
}

impl Searches {
    fn lock(&self) -> PyResult<MutexGuard<'_, State>> {
        self.state.lock().map_err(|_| PyRuntimeError::new_err("a call to these searches panicked, which may have left them broken"))
    }

    //descend's round, its boards added to boards; with columns, a lean round, every board's LEAN_COLUMNS added to them
    #[allow(clippy::too_many_arguments)]
    fn descend_round(
        &self,
        state: &mut State,
        leaves: u32,
        collisions: u32,
        simulations: Option<u32>,
        round: Option<u64>,
        boards: &mut Vec<u8>,
        mut columns: Option<&mut Vec<u16>>,
    ) -> PyResult<()> {
        if let Some(name) = round
            && state.rounds.iter().any(|out| out.name == Some(name))
        {
            return Err(PyValueError::new_err(format!("round {name} is still waiting for its answers")));
        }
        let number = state.rounds_descended;
        state.rounds_descended += 1;
        let mut round_leaves = Vec::new();
        for search in state.searches.iter_mut().filter(|search| !search.stopped) {
            let limits = Round {
                id: number,
                leaves,
                simulations: simulations.unwrap_or(leaves),
                collisions,
                visit_limit: search.simulations,
                memory_limit: self.memory_limit,
            };
            let gathered =
                search.tree.gather(&limits, &self.settings, self.tables.as_ref(), state.cache.as_ref(), boards).map_err(tablebase_error)?;
            if gathered > 0 {
                round_leaves.push((search.id, gathered));
            }
            if let Some(columns) = columns.as_deref_mut() {
                let score = (self.answer_width - 1) as u16;
                for (moves, turn) in search.tree.round_moves(number) {
                    if moves.len() > LEGAL_SLOTS {
                        return Err(PyRuntimeError::new_err(format!(
                            "a leaf with {} legal moves, past a lean row's {LEGAL_SLOTS}",
                            moves.len()
                        )));
                    }
                    columns.extend(moves.iter().map(|&chess_move| self.move_ids.id(chess_move, turn) as u16));
                    columns.extend(std::iter::repeat_n(score, LEAN_COLUMNS - moves.len()));
                }
            }
        }
        state.rounds.push_back(RoundOut { number, name: round, leaves: round_leaves, lean: columns.is_some() });
        Ok(())
    }

    //descend_round in either mode, for the callers that write the boards into an array of their own (room boards long)
    #[allow(clippy::too_many_arguments)]
    fn descend_boards(
        &self,
        room: usize,
        leaves: u32,
        collisions: u32,
        simulations: Option<u32>,
        round: Option<u64>,
        root_of: Option<u64>,
        lean: bool,
    ) -> PyResult<(Vec<u8>, Vec<u16>, Option<RootNumbers>)> {
        let mut state = self.lock()?;
        let searching = if self.concurrent_descents {
            state.shared_searches.iter().filter(|search| !search.stopped).count()
        } else {
            state.searches.iter().filter(|search| !search.stopped).count()
        };
        if leaves as usize * searching > room {
            return Err(PyValueError::new_err(format!("room for {room} boards, not {leaves} for each of {searching} searches")));
        }
        let (mut tokens, mut columns) = (Vec::new(), Vec::new());
        let root = if self.concurrent_descents {
            drop(state);
            self.shared_descend(leaves, collisions, simulations, round, &mut tokens, lean.then_some(&mut columns))?;
            root_of.map(|id| self.shared_root_numbers(id)).transpose()?.flatten()
        } else {
            self.descend_round(&mut state, leaves, collisions, simulations, round, &mut tokens, lean.then_some(&mut columns))?;
            root_of.and_then(|id| root_numbers(&state, id))
        };
        Ok((tokens, columns, root))
    }

    //finish's round
    fn finish_round<N: Number>(&self, state: &mut State, answers: &[N], round: Option<u64>) -> PyResult<Vec<SearchReport>> {
        let index = match round {
            None => (!state.rounds.is_empty()).then_some(0).ok_or_else(|| PyRuntimeError::new_err("no round is waiting for its answers")),
            Some(name) => state
                .rounds
                .iter()
                .position(|out| out.name == Some(name))
                .ok_or_else(|| PyRuntimeError::new_err(format!("round {name} is not waiting for its answers"))),
        }?;
        let boards = state.rounds[index].leaves.iter().map(|&(_, leaves)| leaves as usize).sum();
        let width = if state.rounds[index].lean { LEAN_COLUMNS } else { self.answer_width };
        let answers = answers_for(answers, boards, width)?;
        let RoundOut { number, leaves: round_leaves, lean, .. } = state.rounds.remove(index).expect("the round");
        let index_of: HashMap<u64, usize> = state.searches.iter().enumerate().map(|(index, search)| (search.id, index)).collect();
        let mut rows = answers.chunks_exact(width);
        //a lean answer put back in its columns of a whole row, where finish_leaf reads it
        let mut row = if lean { vec![0.0; self.answer_width] } else { Vec::new() };
        for (id, leaves) in round_leaves {
            let tree = &mut state.searches[index_of[&id]].tree;
            for answer in rows.by_ref().take(leaves as usize) {
                if lean {
                    let (moves, turn) = tree.round_moves(number).next().expect("the round's next leaf");
                    let columns = moves.iter().map(|&chess_move| self.move_ids.id(chess_move, turn));
                    for (column, value) in
                        columns.chain(std::iter::once(self.answer_width - 1)).zip(answer.iter().take(moves.len()).chain(answer.last()))
                    {
                        row[column] = value.number();
                    }
                }
                tree.finish_leaf(
                    number,
                    |column| if lean { row[column] } else { answer[column].number() },
                    self.answer_width,
                    &self.move_ids,
                    state.cache.as_mut(),
                    &self.settings,
                );
            }
        }
        Ok(self.take_reports(state))
    }

    //descend_into's lean round for Searches' own round loop (rounds.rs): the rows written at rows_host, a lane's pinned buffer
    //with room for `room` boards
    #[allow(clippy::too_many_arguments)]
    fn descend_rows(
        &self,
        rows_host: usize,
        room: usize,
        leaves: u32,
        collisions: u32,
        simulations: u32,
        round: u64,
        root_of: u64,
    ) -> PyResult<(usize, Option<RootNumbers>)> {
        let (tokens, columns, root) = self.descend_boards(room, leaves, collisions, Some(simulations), Some(round), Some(root_of), true)?;
        //SAFETY: the lane's rows, room boards long, are the loop's until it launches them
        unsafe { rounds::write_lean_rows(rows_host as *mut u8, &tokens, &columns) };
        Ok((tokens.len() / TOKENS_PER_BOARD, root))
    }

    fn stop_search(&self, state: &mut State, id: u64) -> PyResult<Option<SearchReport>> {
        let search =
            state.searches.iter_mut().find(|search| search.id == id).ok_or_else(|| PyValueError::new_err(format!("no search {id}")))?;
        if search.tree.root().visit_count == 0 && search.tree.pending() == 0 {
            return Err(PyRuntimeError::new_err(format!("search {id} was stopped before its first simulation")));
        }
        search.stopped = true;
        Ok(self.take_reports(state).into_iter().next())
    }

    //removes the searches that are done, the reports of those that were searched
    fn take_reports(&self, state: &mut State) -> Vec<SearchReport> {
        let mut reports = Vec::new();
        for search in state.searches.extract_if(.., |search| search.done()) {
            reports.push(search.report());
            if self.tree_reuse {
                state.kept.insert(search.id, KeptTree { fen: search.fen, moves: search.moves, tree: Kept::Searched(search.tree) });
            }
        }
        reports
    }

    //the id's kept tree moved to the position after `moves`, without the leaves the new root makes stale (search.rs drop_stale_leaves);
    //None if it is another game, lacks the moves, or its root was never expanded
    fn reused_tree(&self, state: &mut State, id: u64, fen: &str, moves: &[String]) -> Option<Tree> {
        let kept = state.kept.remove(&id)?;
        if kept.fen != fen || !moves.starts_with(&kept.moves) {
            kept.tree.abandon();
            return None;
        }
        let since = &moves[kept.moves.len()..];
        let (settings, tables) = (&self.settings, self.tables.as_ref());
        match kept.tree.ready()? {
            Kept::Searched(mut tree) => {
                let (line, _) = line_to(&tree, since)?;
                if !line.is_empty() {
                    tree.play_line(&line);
                    tree.drop_stale_leaves(settings, tables).ok()?;
                }
                Some(tree)
            }
            Kept::Advanced(mut tree, candidates) => {
                let (line, places) = line_to(&tree, since)?;
                tree.move_root(&line);
                match candidates {
                    Some(candidates) => tree.drop_stale_candidates(&places, &candidates, settings, tables).ok()?,
                    None => tree.drop_stale_leaves(settings, tables).ok()?,
                }
                Some(tree)
            }
            Kept::Advancing(..) => unreachable!("ready waits for the thread"),
        }
    }
}

//the edges from the tree's root down the moves (uci), and their places among their nodes' moves; None unless every move has its node
//and the last node has moves
fn line_to(tree: &Tree, moves: &[String]) -> Option<(Vec<usize>, Vec<u16>)> {
    let (mut line, mut places, mut root) = (Vec::new(), Vec::new(), tree.root_index());
    for played in moves {
        let edges = tree.nodes[root].edges();
        let edge = edges.clone().find(|&edge| uci(tree.chess_move(edge)) == *played)?;
        root = tree.child_index(root, edge)?;
        line.push(edge);
        places.push((edge - edges.start) as u16);
    }
    (!tree.nodes[root].edges().is_empty()).then_some((line, places))
}

#[pymethods]
impl Searches {
    //memory_limit_mb (lc0's RamLimitMb, 0 for none) ends a search once its tree's vectors would grow past it; the cache and
    //advance's copy come on top
    #[new]
    #[pyo3(signature = (*, move_id_table, answer_width, c_puct, c_puct_factor, c_puct_base, fpu_reduction, draw, repetitions_for_a_draw,
                        tablebase_folder, tablebase_pieces, tree_reuse=false, cache_positions=0, value_sharpness=1.0, certainty=false,
                        memory_limit_mb=0, advance_threads=4, concurrent_descents=false, transpositions=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        move_id_table: Vec<i16>,
        answer_width: usize,
        c_puct: f64,
        c_puct_factor: f64,
        c_puct_base: f64,
        fpu_reduction: f64,
        draw: f64,
        repetitions_for_a_draw: usize,
        tablebase_folder: Option<&str>,
        tablebase_pieces: usize,
        tree_reuse: bool,
        cache_positions: usize,
        value_sharpness: f64,
        certainty: bool,
        memory_limit_mb: usize,
        advance_threads: usize,
        concurrent_descents: bool,
        transpositions: bool,
    ) -> PyResult<Searches> {
        if value_sharpness <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "value_sharpness multiplies the value's logit, it must be positive, not {value_sharpness}"
            )));
        }
        Ok(Searches {
            settings: SearchSettings {
                c_puct,
                c_puct_factor,
                c_puct_base,
                fpu_reduction,
                draw,
                repetitions_for_a_draw,
                value_sharpness,
                certainty,
                transpositions,
            },
            move_ids: move_ids(move_id_table)?,
            answer_width,
            tables: open_tables(tablebase_folder, tablebase_pieces)?,
            tree_reuse,
            memory_limit: if memory_limit_mb == 0 { usize::MAX } else { memory_limit_mb.saturating_mul(1_000_000) },
            advance_threads: advance_threads.max(1),
            concurrent_descents,
            shared_cache: (concurrent_descents && cache_positions > 0).then(|| SharedCache::new(cache_positions)),
            state: Mutex::new(State {
                searches: Vec::new(),
                rounds: VecDeque::new(),
                rounds_descended: 0,
                kept: HashMap::new(),
                cache: (!concurrent_descents && cache_positions > 0).then(|| Cache::new(cache_positions)),
                shared_searches: Vec::new(),
                shared_kept: HashMap::new(),
            }),
        })
    }

    //the game's moves let the search see repetitions. A mate or stalemate is refused, a position drawn only by a claim or
    //covered by the tables is searched
    fn start(&self, py: Python<'_>, id: u64, fen: &str, moves: Vec<String>, simulations: u32) -> PyResult<()> {
        py.detach(|| {
            if self.concurrent_descents {
                return self.shared_start(id, fen, moves, simulations);
            }
            if simulations == 0 {
                return Err(PyValueError::new_err("a search needs at least one simulation"));
            }
            let mut state = self.lock()?;
            if state.searches.iter().any(|search| search.id == id) {
                return Err(PyValueError::new_err(format!("search {id} is already running")));
            }
            let (board, keys) = play_out(fen, &moves)?;
            if board.legal_moves().is_empty() {
                return Err(PyValueError::new_err(format!("the game is over, there is nothing to search: {fen} {moves:?}")));
            }
            let tree = self
                .reused_tree(&mut state, id, fen, &moves)
                .unwrap_or_else(|| Tree::new(board, keys).with_transpositions(self.settings.transpositions));
            state.searches.push(Search { id, fen: fen.to_string(), moves, simulations, stopped: false, tree });
            Ok(())
        })
    }

    //the boards of up to `leaves` new leaves of every search, in a round of at most `simulations` (default `leaves`); a round given a
    //name (`round`) may be finished by it, in any order
    #[pyo3(signature = (leaves=1, collisions=0, simulations=None, *, round=None))]
    fn descend<'py>(
        &self,
        py: Python<'py>,
        leaves: u32,
        collisions: u32,
        simulations: Option<u32>,
        round: Option<u64>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let boards = py.detach(|| -> PyResult<Vec<u8>> {
            let mut boards = Vec::new();
            if self.concurrent_descents {
                self.shared_descend(leaves, collisions, simulations, round, &mut boards, None)?;
            } else {
                self.descend_round(&mut *self.lock()?, leaves, collisions, simulations, round, &mut boards, None)?;
            }
            Ok(boards)
        })?;
        Ok(PyBytes::new(py, &boards))
    }

    //descend with the boards written into `boards` (a writable array of bytes or int64s with room for `leaves` boards of every search
    //running), and root_visits' numbers of search `root_of` after the round: (the boards written, those numbers, None for no
    //root_of or a search not running).
    //legal_columns: a lean round, into an array of bytes in lean rows, whose finish takes LEAN_COLUMNS numbers a board
    #[pyo3(signature = (boards, leaves=1, collisions=0, simulations=None, *, round=None, root_of=None, legal_columns=false))]
    #[allow(clippy::too_many_arguments)]
    fn descend_into(
        &self,
        py: Python<'_>,
        boards: &Bound<'_, PyAny>,
        leaves: u32,
        collisions: u32,
        simulations: Option<u32>,
        round: Option<u64>,
        root_of: Option<u64>,
        legal_columns: bool,
    ) -> PyResult<(usize, Option<RootNumbers>)> {
        let target = BoardsBuffer::get(boards)?;
        let room = target.boards(if legal_columns { LEAN_ROW } else { TOKENS_PER_BOARD });
        let (tokens, columns, root) =
            py.detach(|| self.descend_boards(room, leaves, collisions, simulations, round, root_of, legal_columns))?;
        target.write(py, &tokens, legal_columns.then_some(&columns[..]))?;
        Ok((tokens.len() / TOKENS_PER_BOARD, root))
    }

    //finishes the round descend named `round` or else the oldest with the network's answers, and returns the reports of the
    //searches that are done
    #[pyo3(signature = (answers, round=None))]
    fn finish(&self, py: Python<'_>, answers: PyBuffer<f32>, round: Option<u64>) -> PyResult<Vec<SearchReport>> {
        let answers = Answers(answer_numbers(py, &answers)?);
        py.detach(move || {
            if self.concurrent_descents {
                return self.shared_finish(answers.numbers(), round);
            }
            self.finish_round(&mut *self.lock()?, answers.numbers(), round)
        })
    }

    //finish, with root_visits' numbers of search `root_of` after it as well (None once the search is done)
    #[pyo3(signature = (answers, round=None, *, root_of))]
    fn finish_with_root(
        &self,
        py: Python<'_>,
        answers: PyBuffer<f32>,
        round: Option<u64>,
        root_of: u64,
    ) -> PyResult<(Vec<SearchReport>, Option<RootNumbers>)> {
        let answers = Answers(answer_numbers(py, &answers)?);
        py.detach(move || {
            if self.concurrent_descents {
                let reports = self.shared_finish(answers.numbers(), round)?;
                return Ok((reports, self.shared_root_numbers(root_of)?));
            }
            let mut guard = self.lock()?;
            let reports = self.finish_round(&mut guard, answers.numbers(), round)?;
            Ok((reports, root_numbers(&guard, root_of)))
        })
    }

    //EngineSearch's rust_rounds: engine_search.py's rounds_on in Rust (rounds.rs), on threads of their own, a network each.
    //networks: (a GraphedBatches of lean lanes, its rust_network()); addresses: the cuda runtime calls'; settings: EngineSearch's
    //rust_round_settings(). RoundLoop.wait has the report
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (networks, addresses, settings, simulations, seconds, extension, only_move, root, round_times))]
    fn round_loop(
        slf: &Bound<'_, Self>,
        networks: Vec<(Py<PyAny>, Bound<'_, PyDict>)>,
        addresses: HashMap<String, usize>,
        settings: rounds::LoopSettings,
        simulations: Option<u32>,
        seconds: Option<f64>,
        extension: Option<(i64, f64)>,
        only_move: bool,
        root: RootNumbers,
        round_times: Vec<f64>,
    ) -> PyResult<rounds::RoundLoop> {
        let searches = Arc::new(slf.clone().unbind());
        rounds::start_loop(searches, networks, addresses, settings, simulations, seconds, extension, only_move, root, round_times)
    }

    //ends a search at its next report: at once when none of its leaves is waiting for the network, else from the finish that
    //brings the last back
    fn stop(&self, py: Python<'_>, id: u64) -> PyResult<Option<SearchReport>> {
        py.detach(|| {
            if self.concurrent_descents {
                return self.shared_stop(id);
            }
            self.stop_search(&mut *self.lock()?, id)
        })
    }

    fn root_visits(&self, py: Python<'_>, id: u64) -> PyResult<RootNumbers> {
        py.detach(|| {
            let numbers = if self.concurrent_descents { self.shared_root_numbers(id)? } else { root_numbers(&*self.lock()?, id) };
            numbers.ok_or_else(|| PyValueError::new_err(format!("no search {id}")))
        })
    }

    //the score a running search would report now (report's)
    fn root_score(&self, py: Python<'_>, id: u64) -> PyResult<f64> {
        py.detach(|| {
            if self.concurrent_descents {
                return self.shared_root_score(id);
            }
            Ok(root_score_of(find_search(&*self.lock()?, id)?))
        })
    }

    //a running search's report as it would be now; None while its root has no moves yet (its first simulation is still out)
    fn report_now(&self, py: Python<'_>, id: u64) -> PyResult<Option<SearchReport>> {
        py.detach(|| {
            if self.concurrent_descents {
                return self.shared_report_now(id);
            }
            let state = self.lock()?;
            let search = find_search(&state, id)?;
            Ok(search.tree.root().has_edges().then(|| search.report()))
        })
    }

    //carries the id's kept tree on to the position after `moves` (the engine's own move last): the root moves down in place, and a
    //thread finds the leaves the next root may make stale and copies its part out, while the opponent thinks. The next start waits for
    //the walk, stops the copy and goes on from the tree as it is. False when no tree is kept for these moves
    fn advance(&self, py: Python<'_>, id: u64, fen: &str, moves: Vec<String>) -> PyResult<bool> {
        py.detach(|| {
            if self.concurrent_descents {
                return self.shared_advance(id, fen, moves);
            }
            let mut state = self.lock()?;
            if state.searches.iter().any(|search| search.id == id) {
                return Err(PyValueError::new_err(format!("search {id} is running")));
            }
            let Some(kept) = state.kept.remove(&id) else { return Ok(false) };
            if kept.fen != fen || !moves.starts_with(&kept.moves) {
                kept.tree.abandon();
                return Ok(false);
            }
            let Some(Kept::Searched(mut tree) | Kept::Advanced(mut tree, _)) = kept.tree.ready() else { return Ok(false) };
            let Some((line, _)) = line_to(&tree, &moves[kept.moves.len()..]) else { return Ok(false) };
            tree.move_root(&line);
            let (stops, threads) = (Arc::new(Stops::default()), self.advance_threads);
            let thread_stops = stops.clone();
            let carry = move || {
                let candidates = tree.stale_candidates(threads, &thread_stops.walk);
                if candidates.is_some() {
                    drop(tree.compact(threads, Some(&thread_stops.copy)));
                }
                (tree, candidates)
            };
            //without a thread the tree is not kept
            let Ok(thread) = std::thread::Builder::new().spawn(carry) else { return Ok(false) };
            state.kept.insert(id, KeptTree { fen: fen.to_string(), moves, tree: Kept::Advancing(thread, stops) });
            Ok(true)
        })
    }

    //uci's ucinewgame: the id's first search starts a fresh tree, and the network's answers are forgotten, as lc0 forgets them
    fn new_game(&self, py: Python<'_>, id: u64) -> PyResult<()> {
        py.detach(|| {
            if self.concurrent_descents {
                return self.shared_new_game(id);
            }
            let mut state = self.lock()?;
            if let Some(kept) = state.kept.remove(&id) {
                kept.tree.abandon();
            }
            if let Some(cache) = state.cache.as_mut() {
                cache.clear();
            }
            Ok(())
        })
    }

    //a running search's root is proven (search.rs root_decided): its move is known, and searching on changes nothing
    fn decided(&self, py: Python<'_>, id: u64) -> PyResult<bool> {
        py.detach(|| {
            if self.concurrent_descents {
                return Ok(self.shared_running(id)?.root_decided());
            }
            Ok(find_search(&*self.lock()?, id)?.tree.root_decided())
        })
    }

    //a running search's root moves not proven lost (search.rs moves_not_lost)
    fn moves_not_lost(&self, py: Python<'_>, id: u64) -> PyResult<usize> {
        py.detach(|| {
            if self.concurrent_descents {
                return Ok(self.shared_running(id)?.moves_not_lost());
            }
            Ok(find_search(&*self.lock()?, id)?.tree.moves_not_lost())
        })
    }

    fn __len__(&self, py: Python<'_>) -> PyResult<usize> {
        py.detach(|| {
            let state = self.lock()?;
            Ok(state.searches.len() + state.shared_searches.len())
        })
    }
}

//finish_leaf's arithmetic, for the tests' python search to use
#[pyfunction]
fn softmax(mut logits: Vec<f32>) -> Vec<f32> {
    search::softmax(&mut logits);
    logits
}

//the cpus this process may run on as the hardware groups them (affinity.rs): each L3's bytes and cpus, and each physical core's
//cpus (its smt siblings)
#[pyfunction]
fn cpu_topology() -> Option<(Vec<(u64, Vec<usize>)>, Vec<Vec<usize>>)> {
    affinity::topology().map(|topology| (topology.l3s, topology.cores))
}

//pins the calling thread to the cpu, one of cpu_topology's; false if it may not run there
#[pyfunction]
fn pin_thread(cpu: usize) -> bool {
    affinity::pin_current(cpu)
}

#[pymodule]
fn selfplay_rs(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<SelfPlayGames>()?;
    module.add_class::<Searches>()?;
    module.add_function(wrap_pyfunction!(softmax, module)?)?;
    module.add_class::<rounds::RoundLoop>()?;
    module.add_function(wrap_pyfunction!(cpu_topology, module)?)?;
    module.add_function(wrap_pyfunction!(pin_thread, module)?)?;
    module.add_function(wrap_pyfunction!(fake_cuda::fake_cuda_addresses, module)?)?;
    module.add_function(wrap_pyfunction!(fake_cuda::fake_cuda_graph, module)?)?;
    module.add("LEAN_COLUMNS", LEAN_COLUMNS)?;
    module.add("LEAN_ROW", LEAN_ROW)?;
    Ok(())
}
