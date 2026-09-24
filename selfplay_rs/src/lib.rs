//mcts.py's search in rust, for many searches at once, while python feeds the network: selfplay.py's games
//(SelfPlayGames, selfplay.RustSelfPlayBatch) and the matches against stockfish (Searches, tools/play/search_match.py).
//A round is descend (every tree walks down to a leaf and the leaves that need the network come back as boards), then
//finish with the network's answers for them
mod games;
mod rules;
mod search;

use games::{FinishedGame, Game, GameSettings};
use pyo3::buffer::{PyBuffer, ReadOnlyCell};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rules::{MOVE_ID_TABLE_LENGTH, MoveIds, Tables, encode_board, final_result, game_result, legal_moves_in_python_order, position_key};
use search::{SearchSettings, Tree};
use shakmaty::fen::Fen;
use shakmaty::uci::UciMove;
use shakmaty::{CastlingMode, Chess, Position};
use shakmaty_syzygy::SyzygyError;

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

//a board from its fen, and the keys of its position and of the positions of the moves played from it
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

//the first half of a round for every tree: the boards of the leaves that need the network, TOKENS_PER_BOARD bytes each,
//and in waiting the trees they belong to
fn descend_all<'a>(
    trees: impl Iterator<Item = &'a mut Tree>,
    settings: &SearchSettings,
    tables: Option<&Tables>,
    waiting: &mut Vec<usize>,
) -> PyResult<Vec<u8>> {
    if !waiting.is_empty() {
        return Err(PyRuntimeError::new_err("the last leaves are still waiting for finish"));
    }
    let mut boards = Vec::new();
    for (index, tree) in trees.enumerate() {
        if tree.descend(settings, tables, &mut boards).map_err(tablebase_error)? {
            waiting.push(index);
        }
    }
    Ok(boards)
}

//the network's answers (float32, answer_width a board) for the boards descend gave it
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
    //every game starts from here: the board, the keys of the positions played to reach it, and how many
    start: Chess,
    start_keys: Vec<u64>,
    start_plies: usize,
    games: Vec<Game>,
    waiting: Vec<usize>, //the games whose leaf the network is evaluating, in the order of their boards
    finished: Vec<FinishedGame>,
}

#[pymethods]
impl SelfPlayGames {
    #[new]
    #[pyo3(signature = (*, move_id_table, answer_width, start_fen, start_moves, seed, simulations, sampled_plies, max_plies, tree_reuse,
                        top_moves, no_move, c_puct, c_puct_factor, c_puct_base, fpu_reduction, draw, repetitions_for_a_draw,
                        root_noise_alpha, root_noise_share, tablebase_folder, tablebase_pieces))]
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
    ) -> PyResult<SelfPlayGames> {
        let (start, start_keys) = play_out(start_fen, &start_moves)?;
        Ok(SelfPlayGames {
            settings: GameSettings {
                search: SearchSettings { c_puct, c_puct_factor, c_puct_base, fpu_reduction, draw, repetitions_for_a_draw },
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
        })
    }

    //starts game number `number`: True while it is played, False when it was over before it began (it is then among
    //the finished, without positions)
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

    //the first half of a round: the boards of the leaves that need the network
    fn descend<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let trees = self.games.iter_mut().map(|game| &mut game.tree);
        let boards = descend_all(trees, &self.settings.search, self.tables.as_ref(), &mut self.waiting)?;
        Ok(PyBytes::new(py, &boards))
    }

    //the second half: the network's answers for descend's boards finish their simulations, and every game whose root
    //has its simulations plays a move. Returns how many games ended
    fn finish(&mut self, py: Python<'_>, answers: PyBuffer<f32>) -> PyResult<usize> {
        let answers = answers_for(py, &answers, self.waiting.len(), self.answer_width)?;
        for (row, &index) in self.waiting.iter().enumerate() {
            let answer = &answers[row * self.answer_width..(row + 1) * self.answer_width];
            self.games[index].tree.finish_leaf(|column| answer[column].get(), self.answer_width, &self.move_ids);
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

    //what the rules here say about the position after `moves` from the start board, for the tests to hold against the
    //python: its encoding, its legal moves' ids in their order, mcts.game_result and selfplay.final_result
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

//one search of a match: mcts.search from a position of a game, without noise
struct Search {
    id: u64,
    simulations: u32,
    tree: Tree,
}

//what a search found, as player.search_move reports it: the line it expects (its move first) as uci moves, the mean score
//of that move for the side to move, and every root move with its visits, the most visited first
type SearchReport = (u64, Vec<String>, f64, Vec<(String, u32)>);

#[pyclass(unsendable)]
struct Searches {
    settings: SearchSettings,
    move_ids: MoveIds,
    answer_width: usize,
    tables: Option<Tables>,
    searches: Vec<Search>,
    waiting: Vec<usize>, //the searches whose leaf the network is evaluating, in the order of their boards
}

#[pymethods]
impl Searches {
    #[new]
    #[pyo3(signature = (*, move_id_table, answer_width, c_puct, c_puct_factor, c_puct_base, fpu_reduction, draw, repetitions_for_a_draw,
                        tablebase_folder, tablebase_pieces))]
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
    ) -> PyResult<Searches> {
        Ok(Searches {
            settings: SearchSettings { c_puct, c_puct_factor, c_puct_base, fpu_reduction, draw, repetitions_for_a_draw },
            move_ids: move_ids(move_id_table)?,
            answer_width,
            tables: open_tables(tablebase_folder, tablebase_pieces)?,
            searches: Vec::new(),
            waiting: Vec::new(),
        })
    }

    //starts a search of `simulations` from the position after `moves` from `fen`, the game's moves so that the search
    //sees its repetitions. Like mcts.search it refuses a position whose game is over or that the tablebase knows
    fn start(&mut self, id: u64, fen: &str, moves: Vec<String>, simulations: u32) -> PyResult<()> {
        if simulations == 0 {
            return Err(PyValueError::new_err("a search needs at least one simulation"));
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
        self.searches.push(Search { id, simulations, tree: Tree::new(board, keys) });
        Ok(())
    }

    //the first half of a round: the boards of the leaves that need the network
    fn descend<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let trees = self.searches.iter_mut().map(|search| &mut search.tree);
        let boards = descend_all(trees, &self.settings, self.tables.as_ref(), &mut self.waiting)?;
        Ok(PyBytes::new(py, &boards))
    }

    //the second half: the network's answers for descend's boards finish their simulations. Returns what every search that
    //now has all its simulations found, as (id, line, score, root moves)
    fn finish(&mut self, py: Python<'_>, answers: PyBuffer<f32>) -> PyResult<Vec<SearchReport>> {
        let answers = answers_for(py, &answers, self.waiting.len(), self.answer_width)?;
        for (row, &index) in self.waiting.iter().enumerate() {
            let answer = &answers[row * self.answer_width..(row + 1) * self.answer_width];
            self.searches[index].tree.finish_leaf(|column| answer[column].get(), self.answer_width, &self.move_ids);
        }
        self.waiting.clear();

        let mut reports = Vec::new();
        self.searches.retain(|search| {
            let tree = &search.tree;
            if tree.root().visit_count < search.simulations {
                return true;
            }
            let line = tree.principal_variation();
            let chosen = &tree.nodes[line[0]];
            //the chosen move's mean score, or when it was never visited (one simulation) the root's own for its side to move
            let score = if chosen.visit_count > 0 { chosen.mean_score() } else { 1.0 - tree.root().mean_score() };
            let moves =
                |nodes: &[usize]| -> Vec<String> { nodes.iter().map(|&node| uci(tree.nodes[node].chess_move.expect("a child"))).collect() };
            let root_moves = tree
                .children_by_visits()
                .into_iter()
                .map(|child| (uci(tree.nodes[child].chess_move.expect("a child")), tree.nodes[child].visit_count));
            reports.push((search.id, moves(&line), score, root_moves.collect()));
            false
        });
        Ok(reports)
    }

    //the searches still running
    fn __len__(&self) -> usize {
        self.searches.len()
    }
}

//the numbers of finish_leaf's priors and score, for the tests' python search to use the same
#[pyfunction]
fn softmax(logits: Vec<f32>) -> Vec<f32> {
    let mut priors = logits;
    search::softmax(&mut priors);
    priors
}

#[pyfunction]
fn sigmoid(logit: f32) -> f32 {
    search::sigmoid(logit)
}

#[pymodule]
fn selfplay_rs(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<SelfPlayGames>()?;
    module.add_class::<Searches>()?;
    module.add_function(wrap_pyfunction!(softmax, module)?)?;
    module.add_function(wrap_pyfunction!(sigmoid, module)?)?;
    Ok(())
}
