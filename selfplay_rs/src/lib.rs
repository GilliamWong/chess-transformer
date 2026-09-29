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
                search: SearchSettings::mcts(c_puct, c_puct_factor, c_puct_base, fpu_reduction, draw, repetitions_for_a_draw),
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
            let settings = &self.settings.search;
            self.games[index].tree.finish_leaf(|column| answer[column].get(), self.answer_width, &self.move_ids, settings, self.cache.as_mut());
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

//one search of a match: mcts.search, without noise. It ends at its simulations or when stopped
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
        (self.stopped || self.tree.root().visit_count >= self.simulations) && self.tree.pending() == 0
    }

    //as player.search_move reports it
    fn report(&self, settings: &SearchSettings) -> SearchReport {
        let tree = &self.tree;
        let line = tree.principal_variation(settings);
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

//the tree of a game's search that ended last, for the game's next search
struct KeptTree {
    fen: String,
    moves: Vec<String>,
    tree: Tree,
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
}

impl Searches {
    //removes the searches that are done, the reports of those that were searched
    fn take_reports(&mut self) -> Vec<SearchReport> {
        let mut reports = Vec::new();
        for search in self.searches.extract_if(.., |search| search.done()) {
            reports.push(search.report(&self.settings));
            if self.tree_reuse {
                self.kept.insert(search.id, KeptTree { fen: search.fen, moves: search.moves, tree: search.tree });
            }
        }
        reports
    }

    //the id's kept tree carried on to the position after `moves`, when it is the same game and holds the moves played since
    //and was searched with the same contempt: its draws' scores and their proofs are that contempt's
    fn reused_tree(&mut self, id: u64, fen: &str, moves: &[String], contempt: f64) -> Option<Tree> {
        let kept = self.kept.remove(&id)?;
        if kept.fen != fen || !moves.starts_with(&kept.moves) || kept.tree.contempt != contempt {
            return None;
        }
        let mut tree = kept.tree;
        for played in &moves[kept.moves.len()..] {
            let edge = tree.root().edges().find(|&edge| uci(tree.chess_move(edge)) == *played)?;
            tree.play(edge, true);
        }
        Some(tree)
    }
}

#[pymethods]
impl Searches {
    //tree_reuse keeps the tree of a game's (an id's) last search for its next search; cache_positions is the size of
    //the cache of network answers by position (search.rs Cache), 0 for none; the settings after it are SearchSettings'
    //beyond mcts.py's constants, their defaults mcts.py's search
    #[new]
    #[pyo3(signature = (*, move_id_table, answer_width, c_puct, c_puct_factor, c_puct_base, fpu_reduction, draw, repetitions_for_a_draw,
                        tablebase_folder, tablebase_pieces, tree_reuse=false, cache_positions=0, policy_temperature=1.0, certainty=false,
                        fifty_move_horizon=0.0, cpuct_stdev_scale=0.0, cpuct_stdev_prior=0.2, cpuct_stdev_prior_weight=2.0, lcb_stdevs=0.0,
                        lcb_least_share=0.1, opponent_c_puct_scale=1.0, opponent_fpu_scale=1.0,
                        transposition_visits=0, uncertainty_scale=0.0))]
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
        policy_temperature: f64,
        certainty: bool,
        fifty_move_horizon: f64,
        cpuct_stdev_scale: f64,
        cpuct_stdev_prior: f64,
        cpuct_stdev_prior_weight: f64,
        lcb_stdevs: f64,
        lcb_least_share: f64,
        opponent_c_puct_scale: f64,
        opponent_fpu_scale: f64,
        transposition_visits: u32,
        uncertainty_scale: f64,
    ) -> PyResult<Searches> {
        if policy_temperature <= 0.0 {
            return Err(PyValueError::new_err(format!("a policy temperature must be above 0, not {policy_temperature}")));
        }
        if fifty_move_horizon < 0.0 {
            return Err(PyValueError::new_err(format!("a fifty move horizon must be 0 (none) or above, not {fifty_move_horizon}")));
        }
        if uncertainty_scale < 0.0 {
            return Err(PyValueError::new_err(format!("an uncertainty scale must be 0 (none) or above, not {uncertainty_scale}")));
        }
        if cpuct_stdev_prior <= 0.0 || cpuct_stdev_prior_weight < 0.0 {
            return Err(PyValueError::new_err("the stdev prior must be above 0 and its weight at least 0"));
        }
        Ok(Searches {
            settings: SearchSettings {
                policy_temperature,
                certainty,
                fifty_move_horizon,
                cpuct_stdev_scale,
                cpuct_stdev_prior,
                cpuct_stdev_prior_weight,
                lcb_stdevs,
                lcb_least_share,
                opponent_c_puct_scale,
                opponent_fpu_scale,
                transposition_visits,
                uncertainty_scale,
                ..SearchSettings::mcts(c_puct, c_puct_factor, c_puct_base, fpu_reduction, draw, repetitions_for_a_draw)
            },
            move_ids: move_ids(move_id_table)?,
            answer_width,
            tables: open_tables(tablebase_folder, tablebase_pieces)?,
            searches: Vec::new(),
            rounds: VecDeque::new(),
            tree_reuse,
            kept: HashMap::new(),
            cache: (cache_positions > 0).then(|| Cache::new(cache_positions)),
        })
    }

    //the game's moves let the search see repetitions. Like mcts.search it refuses a finished or tablebase position.
    //With tree reuse, the tree of this id's last search is kept when it holds the moves played since and had the same
    //contempt. contempt is taken off a draw's score for the side to move here and added for the other side (search.rs
    //Tree), 0 is mcts.py's search
    #[pyo3(signature = (id, fen, moves, simulations, contempt=0.0))]
    fn start(&mut self, id: u64, fen: &str, moves: Vec<String>, simulations: u32, contempt: f64) -> PyResult<()> {
        if simulations == 0 {
            return Err(PyValueError::new_err("a search needs at least one simulation"));
        }
        //a draw must stay between a loss and a win, as certainty's proofs tell them apart by their scores
        if contempt.is_nan() || contempt.abs() >= self.settings.draw.min(1.0 - self.settings.draw) {
            return Err(PyValueError::new_err(format!("a contempt of {contempt} makes a draw a loss or a win")));
        }
        if self.searches.iter().any(|search| search.id == id) {
            return Err(PyValueError::new_err(format!("search {id} is already running")));
        }
        let (board, keys) = play_out(fen, &moves)?;
        let settings = &self.settings;
        let over = game_result(&board, &keys, &board.legal_moves(), settings.draw, settings.repetitions_for_a_draw, self.tables.as_ref())
            .map_err(tablebase_error)?;
        if over.is_some() {
            return Err(PyValueError::new_err(format!(
                "the game is over or the tablebase knows it, there is nothing to search: {fen} {moves:?}"
            )));
        }
        let mut tree = self.reused_tree(id, fen, &moves, contempt).unwrap_or_else(|| Tree::new(board, keys));
        tree.contempt = contempt;
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
            let limits = Round { leaves, simulations: simulations.unwrap_or(leaves), collisions, visit_limit: search.simulations };
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
                tree.finish_leaf(|column| answer[column].get(), self.answer_width, &self.move_ids, &self.settings, self.cache.as_mut());
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

    //the move a running search would play now (chosen_edge) and its mean score, None before the root's first visit
    //to a move: for time management, which spends more while it keeps changing or its score falls
    fn leader(&self, id: u64) -> PyResult<Option<(String, f64)>> {
        let search = self.searches.iter().find(|search| search.id == id).ok_or_else(|| PyValueError::new_err(format!("no search {id}")))?;
        let tree = &search.tree;
        if tree.root().edges().is_empty() {
            return Ok(None);
        }
        let edge = tree.chosen_edge(&self.settings);
        Ok(tree.child(edge).filter(|child| child.visit_count > 0).map(|child| (uci(tree.chess_move(edge)), child.mean_score())))
    }

    //with certainty: the running search has proven its root's result (a move wins, or every move's result is known),
    //so more simulations change nothing
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
