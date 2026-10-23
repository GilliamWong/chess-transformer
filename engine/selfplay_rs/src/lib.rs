//mcts.py's search for many trees at once while python runs the network: self play games (SelfPlayGames) and match
//searches (Searches). A round is descend (the leaves that need the network come back as boards), then finish
mod games;
mod rules;
mod search;

use games::{FinishedGame, Game, GameSettings};
use pyo3::buffer::{PyBuffer, ReadOnlyCell};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rules::{MOVE_ID_TABLE_LENGTH, MoveIds, Tables, encode_board, final_result, game_result, legal_moves_in_python_order, position_key};
use search::{Cache, Round, SearchSettings, Tree};
use shakmaty::fen::Fen;
use shakmaty::uci::UciMove;
use shakmaty::{CastlingMode, Chess, Position};
use shakmaty_syzygy::SyzygyError;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
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

//the network's answers, answer_width float32s for each board descend gave it
fn answers_for<'a>(py: Python<'a>, answers: &'a PyBuffer<f32>, boards: usize, answer_width: usize) -> PyResult<&'a [ReadOnlyCell<f32>]> {
    let answers = answers.as_slice(py).ok_or_else(|| PyValueError::new_err("the answers must be one contiguous array"))?;
    if answers.len() != boards * answer_width {
        return Err(PyValueError::new_err(format!("{} numbers of answers for {boards} boards of {answer_width} each", answers.len())));
    }
    Ok(answers)
}

fn uci(chess_move: shakmaty::Move) -> String {
    chess_move.to_uci(CastlingMode::Standard).to_string()
}

#[pyclass(unsendable)]
struct SelfPlayGames {
    settings: GameSettings,
    move_ids: MoveIds,
    answer_width: usize,
    tables: Option<Tables>,
    seed: u64,
    //every game starts here, with the keys of the positions played to reach it
    start: Chess,
    start_keys: Vec<u64>,
    start_plies: usize,
    games: Vec<Game>,
    waiting: Vec<usize>, //games whose leaf the network is evaluating, in board order
    finished: Vec<FinishedGame>,
    cache: Option<Cache>, //shared by the games: the network answers a position the same in every game
    simulations_per_round: u32,
}

#[pymethods]
impl SelfPlayGames {
    //cache_positions is the size of the cache of network answers by position (search.rs Cache), 0 for none.
    //simulations_per_round: a game's round may do this many simulations, all but the last answered at once (by the
    //cache or the game's end); the games are the same, in fewer rounds
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

    //the boards of the leaves that need the network, at most one a game (TOKENS_PER_BOARD bytes each). As in
    //reference_selfplay a move gets at least one simulation, and the move is played once the root has its simulations
    //(so a round stops there); until the root's noise is in, a round is one simulation
    fn descend<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        if !self.waiting.is_empty() {
            return Err(PyRuntimeError::new_err("the last leaves are still waiting for finish"));
        }
        let mut boards = Vec::new();
        for (index, game) in self.games.iter_mut().enumerate() {
            let round = Round {
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
        let answers = answers_for(py, &answers, self.waiting.len(), self.answer_width)?;
        for (row, &index) in self.waiting.iter().enumerate() {
            let answer = &answers[row * self.answer_width..(row + 1) * self.answer_width];
            self.games[index].tree.finish_leaf(
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

    //games being played or finished and not yet taken
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

//one search of a match: mcts.search, without noise. It ends at its simulations, when stopped, or when its tree has no
//room for more (search.rs make_room)
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
        let score = match tree.child(line[0]) {
            Some(chosen) if chosen.visit_count > 0 => chosen.mean_score(),
            _ => 1.0 - tree.root().mean_score(),
        };
        let moves = |edges: &[usize]| -> Vec<String> { edges.iter().map(|&edge| uci(tree.chess_move(edge))).collect() };
        let root_moves = tree.edges_by_visits().into_iter().map(|edge| (uci(tree.chess_move(edge)), tree.visits(edge)));
        (self.id, moves(&line), score, root_moves.collect(), tree.root().visit_count)
    }
}

//(id, expected line in uci, the chosen move's mean score, root moves by visits, the root's visits)
type SearchReport = (u64, Vec<String>, f64, Vec<(String, u32)>, u32);

//the tree of a game's search that ended last, for the game's next search, at the position after `moves`
struct KeptTree {
    fen: String,
    moves: Vec<String>,
    tree: Kept,
}

//a kept tree as its last search left it, or carried on by advance: on its thread, or back from it with the leaves the
//next root may make stale (None if the thread was stopped before it found them all)
enum Kept {
    Searched(Tree),
    Advancing(JoinHandle<(Tree, Option<Vec<Vec<u16>>>)>, Arc<AtomicBool>),
    Advanced(Tree, Option<Vec<Vec<u16>>>),
}

impl Kept {
    //the tree to search on: an advance not done yet is stopped where it is, so that it takes none of the move's time
    fn ready(self) -> Option<Kept> {
        match self {
            Kept::Advancing(thread, abort) => {
                abort.store(true, Ordering::Relaxed);
                thread.join().ok().map(|(tree, candidates)| Kept::Advanced(tree, candidates))
            }
            kept => Some(kept),
        }
    }

    //no longer wanted: an advance's thread stops at its next look and frees the tree
    fn abandon(self) {
        if let Kept::Advancing(_, abort) = self {
            abort.store(true, Ordering::Relaxed);
        }
    }
}

#[pyclass(unsendable)]
struct Searches {
    settings: SearchSettings,
    move_ids: MoveIds,
    answer_width: usize,
    tables: Option<Tables>,
    searches: Vec<Search>,
    //the rounds whose leaves the network is evaluating, oldest first: each search's id and its leaves, in board order
    rounds: VecDeque<Vec<(u64, u32)>>,
    tree_reuse: bool,
    //by id: a new game under an id drops the last game's tree at its first search
    kept: HashMap<u64, KeptTree>,
    cache: Option<Cache>,
    memory_limit: usize, //bytes a search's tree may take (search.rs Round), usize::MAX for no limit
}

impl Searches {
    //removes the searches that are done, the reports of those that were searched
    fn take_reports(&mut self) -> Vec<SearchReport> {
        let mut reports = Vec::new();
        for search in self.searches.extract_if(.., |search| search.done()) {
            reports.push(search.report());
            if self.tree_reuse {
                self.kept.insert(search.id, KeptTree { fen: search.fen, moves: search.moves, tree: Kept::Searched(search.tree) });
            }
        }
        reports
    }

    //the id's kept tree carried on to the position after `moves`, when it is the same game and holds the moves played
    //since, without the leaves the new root makes stale (search.rs drop_stale_leaves). A tree that advance carried on
    //moves its root down in place, and takes out the stale leaves among the candidates advance found (or walks for
    //them, if it was stopped first). A root the last search never expanded (a repeat it scored as a draw) starts over
    fn reused_tree(&mut self, id: u64, fen: &str, moves: &[String]) -> Option<Tree> {
        let kept = self.kept.remove(&id)?;
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

//the moves since (uci) as the edges from the tree's root down to the position after them, and as their places among
//their nodes' moves, when every move has its node and the last node has moves (a search reports from its root's)
fn line_to(tree: &Tree, moves: &[String]) -> Option<(Vec<usize>, Vec<u16>)> {
    let (mut line, mut places, mut root) = (Vec::new(), Vec::new(), tree.root());
    for played in moves {
        let edges = root.edges();
        let edge = edges.clone().find(|&edge| uci(tree.chess_move(edge)) == *played)?;
        root = tree.child(edge)?;
        line.push(edge);
        places.push((edge - edges.start) as u16);
    }
    (!root.edges().is_empty()).then_some((line, places))
}

#[pymethods]
impl Searches {
    //tree_reuse keeps the tree of a game's (an id's) last search for its next search; cache_positions is the size of
    //the cache of network answers by position (search.rs Cache), 0 for none; value_sharpness see search.rs sharpen,
    //certainty search.rs SearchSettings'. memory_limit_mb (lc0's RamLimitMb, 0 for none) ends a search once its tree's
    //vectors would grow past that many megabytes; the cache comes on top, and so does the copy a kept tree's next
    //search makes of the part it keeps
    #[new]
    #[pyo3(signature = (*, move_id_table, answer_width, c_puct, c_puct_factor, c_puct_base, fpu_reduction, draw, repetitions_for_a_draw,
                        tablebase_folder, tablebase_pieces, tree_reuse=false, cache_positions=0, value_sharpness=1.0, certainty=false,
                        memory_limit_mb=0))]
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
    ) -> PyResult<Searches> {
        if value_sharpness <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "value_sharpness multiplies the value's logit, it must be positive, not {value_sharpness}"
            )));
        }
        Ok(Searches {
            settings: SearchSettings { c_puct, c_puct_factor, c_puct_base, fpu_reduction, draw, repetitions_for_a_draw, value_sharpness, certainty },
            move_ids: move_ids(move_id_table)?,
            answer_width,
            tables: open_tables(tablebase_folder, tablebase_pieces)?,
            searches: Vec::new(),
            rounds: VecDeque::new(),
            tree_reuse,
            kept: HashMap::new(),
            cache: (cache_positions > 0).then(|| Cache::new(cache_positions)),
            memory_limit: if memory_limit_mb == 0 { usize::MAX } else { memory_limit_mb.saturating_mul(1_000_000) },
        })
    }

    //the game's moves let the search see repetitions. It refuses a mate or stalemate, but searches a position drawn only
    //by a claim (threefold, fifty moves), which a gui may still ask a move for, and one the tables cover (search.rs
    //gather). With tree reuse, the tree of this id's last search is kept when it holds the moves played since
    fn start(&mut self, id: u64, fen: &str, moves: Vec<String>, simulations: u32) -> PyResult<()> {
        if simulations == 0 {
            return Err(PyValueError::new_err("a search needs at least one simulation"));
        }
        if self.searches.iter().any(|search| search.id == id) {
            return Err(PyValueError::new_err(format!("search {id} is already running")));
        }
        let (board, keys) = play_out(fen, &moves)?;
        if board.legal_moves().is_empty() {
            return Err(PyValueError::new_err(format!("the game is over, there is nothing to search: {fen} {moves:?}")));
        }
        let tree = self.reused_tree(id, fen, &moves).unwrap_or_else(|| Tree::new(board, keys));
        self.searches.push(Search { id, fen: fen.to_string(), moves, simulations, stopped: false, tree });
        Ok(())
    }

    //the boards of up to `leaves` new leaves of every search for the network, in a round of at most `simulations`
    //(default `leaves`) with those finished at once (see search.rs gather). Rounds may be gathered before the last ones
    //are finished, so that the network is never idle
    #[pyo3(signature = (leaves=1, collisions=0, simulations=None))]
    fn descend<'py>(&mut self, py: Python<'py>, leaves: u32, collisions: u32, simulations: Option<u32>) -> PyResult<Bound<'py, PyBytes>> {
        let mut boards = Vec::new();
        let mut round = Vec::new();
        for search in self.searches.iter_mut().filter(|search| !search.stopped) {
            let limits = Round {
                leaves,
                simulations: simulations.unwrap_or(leaves),
                collisions,
                visit_limit: search.simulations,
                memory_limit: self.memory_limit,
            };
            let gathered = search
                .tree
                .gather(&limits, &self.settings, self.tables.as_ref(), self.cache.as_ref(), &mut boards)
                .map_err(tablebase_error)?;
            if gathered > 0 {
                round.push((search.id, gathered));
            }
        }
        self.rounds.push_back(round);
        Ok(PyBytes::new(py, &boards))
    }

    //finishes the oldest round with the network's answers, and returns the reports of the searches that are done
    fn finish(&mut self, py: Python<'_>, answers: PyBuffer<f32>) -> PyResult<Vec<SearchReport>> {
        let round = self.rounds.pop_front().ok_or_else(|| PyRuntimeError::new_err("no round is waiting for its answers"))?;
        let boards = round.iter().map(|&(_, leaves)| leaves as usize).sum();
        let answers = answers_for(py, &answers, boards, self.answer_width)?;
        let index_of: HashMap<u64, usize> = self.searches.iter().enumerate().map(|(index, search)| (search.id, index)).collect();
        let mut rows = answers.chunks_exact(self.answer_width);
        for (id, leaves) in round {
            let tree = &mut self.searches[index_of[&id]].tree;
            for answer in rows.by_ref().take(leaves as usize) {
                tree.finish_leaf(|column| answer[column].get(), self.answer_width, &self.move_ids, self.cache.as_mut(), &self.settings);
            }
        }
        Ok(self.take_reports())
    }

    //ends a search at its next report: at once when none of its leaves is waiting for the network, else from the finish
    //that brings the last of them back
    fn stop(&mut self, id: u64) -> PyResult<Option<SearchReport>> {
        let search =
            self.searches.iter_mut().find(|search| search.id == id).ok_or_else(|| PyValueError::new_err(format!("no search {id}")))?;
        if search.tree.root().visit_count == 0 && search.tree.pending() == 0 {
            return Err(PyRuntimeError::new_err(format!("search {id} was stopped before its first simulation")));
        }
        search.stopped = true;
        Ok(self.take_reports().into_iter().next())
    }

    //(the root's visits, the visits of its two most visited moves) of a running search, for smart pruning: the most
    //visited move is decided once the simulations left cannot close the gap
    fn root_visits(&self, id: u64) -> PyResult<(u32, u32, u32)> {
        let search = self.searches.iter().find(|search| search.id == id).ok_or_else(|| PyValueError::new_err(format!("no search {id}")))?;
        let (most, second) = search.tree.two_most_visits();
        Ok((search.tree.root().visit_count, most, second))
    }

    //the score a running search would report now (report's), for a clock that compares it with the last search's
    fn root_score(&self, id: u64) -> PyResult<f64> {
        let search = self.searches.iter().find(|search| search.id == id).ok_or_else(|| PyValueError::new_err(format!("no search {id}")))?;
        let tree = &search.tree;
        Ok(match tree.principal_variation().first().and_then(|&edge| tree.child(edge)) {
            Some(chosen) if chosen.visit_count > 0 => chosen.mean_score(),
            _ => 1.0 - tree.root().mean_score(),
        })
    }

    //carries the id's kept tree on to the position after `moves` (the game's, the engine's own move last), for uci.py to
    //call once that move is sent. The root moves down in place, and on a thread of its own, while the opponent thinks,
    //the tree finds the leaves the next root may make stale (search.rs stale_candidates) and copies the root's part out
    //of the rest (compact). The next search's start then only moves the root down the reply; one that comes before the
    //thread is done stops it and goes on from the tree as far as it got. False when no tree is kept for these moves
    fn advance(&mut self, id: u64, fen: &str, moves: Vec<String>) -> PyResult<bool> {
        if self.searches.iter().any(|search| search.id == id) {
            return Err(PyValueError::new_err(format!("search {id} is running")));
        }
        let Some(kept) = self.kept.remove(&id) else { return Ok(false) };
        if kept.fen != fen || !moves.starts_with(&kept.moves) {
            kept.tree.abandon();
            return Ok(false);
        }
        let Some(Kept::Searched(mut tree) | Kept::Advanced(mut tree, _)) = kept.tree.ready() else { return Ok(false) };
        let Some((line, _)) = line_to(&tree, &moves[kept.moves.len()..]) else { return Ok(false) };
        tree.move_root(&line);
        let abort = Arc::new(AtomicBool::new(false));
        let stop = abort.clone();
        let carry = move || {
            let candidates = tree.stale_candidates(&stop);
            if candidates.is_some() {
                drop(tree.compact(Some(&stop)));
            }
            (tree, candidates)
        };
        //without a thread the tree is not kept
        let Ok(thread) = std::thread::Builder::new().spawn(carry) else { return Ok(false) };
        self.kept.insert(id, KeptTree { fen: fen.to_string(), moves, tree: Kept::Advancing(thread, abort) });
        Ok(true)
    }

    //a new game under the id (uci's ucinewgame): its first search starts a fresh tree, and the network's answers are
    //forgotten, as lc0 forgets them
    fn new_game(&mut self, id: u64) {
        if let Some(kept) = self.kept.remove(&id) {
            kept.tree.abandon();
        }
        if let Some(cache) = self.cache.as_mut() {
            cache.clear();
        }
    }

    //a running search's root is proven (search.rs root_decided): its move is known, and searching on changes nothing
    fn decided(&self, id: u64) -> PyResult<bool> {
        let search = self.searches.iter().find(|search| search.id == id).ok_or_else(|| PyValueError::new_err(format!("no search {id}")))?;
        Ok(search.tree.root_decided())
    }

    //the searches still running
    fn __len__(&self) -> usize {
        self.searches.len()
    }
}

//finish_leaf's arithmetic, for the tests' python search to use
#[pyfunction]
fn softmax(logits: Vec<f32>) -> Vec<f32> {
    let mut priors = logits;
    search::softmax(&mut priors);
    priors
}

#[pymodule]
fn selfplay_rs(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<SelfPlayGames>()?;
    module.add_class::<Searches>()?;
    module.add_function(wrap_pyfunction!(softmax, module)?)?;
    Ok(())
}
