//the games of tests/reference_selfplay.py's SelfPlayBatch: without the random numbers, the python's move for move
use crate::rules::{MoveIds, Tables, encode_board, final_result};
use crate::search::{SearchSettings, Tree};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use shakmaty::{Chess, Color, Position};
use shakmaty_syzygy::SyzygyError;

//the search's constants from mcts.py and the game's from selfplay.py
pub struct GameSettings {
    pub search: SearchSettings,
    pub simulations: u32,
    pub sampled_plies: usize,
    pub max_plies: usize,
    pub tree_reuse: bool,
    pub top_moves: usize,
    pub no_move: i16,
    pub root_noise_alpha: f64,
    pub root_noise_share: f64,
}

//selfplay.Game: every position before a move with what the search found there
#[derive(Default)]
pub struct Record {
    pub boards: Vec<u8>,
    pub visit_moves: Vec<i16>,
    pub visit_shares: Vec<f32>,
    pub root_scores: Vec<f32>,
    pub white_to_move: Vec<u8>,
}

pub struct FinishedGame {
    pub number: u64,
    pub record: Record,
    pub white_score: f64,
}

//reference_selfplay.GameInProgress
pub struct Game {
    number: u64,
    random: ChaCha8Rng,
    pub tree: Tree,
    plies: usize, //len(board.move_stack)
    noised: bool,
    record: Record,
}

impl Game {
    pub fn new(number: u64, seed: u64, board: &Chess, keys: &[u64], plies: usize) -> Game {
        let mut seed_bytes = [0; 32];
        seed_bytes[..8].copy_from_slice(&seed.to_le_bytes());
        seed_bytes[8..16].copy_from_slice(&number.to_le_bytes());
        Game {
            number,
            random: ChaCha8Rng::from_seed(seed_bytes),
            tree: Tree::new(board.clone(), keys.to_vec()),
            plies,
            noised: false,
            record: Record::default(),
        }
    }

    pub fn final_result(&self, max_plies: usize, tables: Option<&Tables>) -> Result<Option<f64>, SyzygyError> {
        final_result(&self.tree.board, &self.tree.keys, self.plies, max_plies, tables)
    }

    pub fn into_finished(self, white_score: f64) -> FinishedGame {
        FinishedGame { number: self.number, record: self.record, white_score }
    }

    //the root's noise is still to come: it comes once the root is expanded, before the next simulation
    pub fn awaits_noise(&self) -> bool {
        !self.noised
    }

    //GameInProgress.add_noise_once_expanded
    pub fn add_noise_once_expanded(&mut self, settings: &GameSettings) {
        if !self.noised && !self.tree.root().edges().is_empty() {
            self.tree.add_root_noise(&mut self.random, settings.root_noise_alpha, settings.root_noise_share);
            self.noised = true;
        }
    }

    //GameInProgress.play_move
    pub fn play_move(&mut self, settings: &GameSettings, move_ids: &MoveIds) {
        let root = *self.tree.root();
        let turn = self.tree.board.turn();
        //the most visited first, ties in the children's order (np.argsort's stable sort)
        let most_visited: Vec<usize> =
            self.tree.edges_by_visits().into_iter().take(settings.top_moves).filter(|&edge| self.tree.visits(edge) > 0).collect();
        let visits: u32 = most_visited.iter().map(|&edge| self.tree.visits(edge)).sum();
        for slot in 0..settings.top_moves {
            match most_visited.get(slot) {
                Some(&edge) => {
                    self.record.visit_moves.push(move_ids.id(self.tree.chess_move(edge), turn) as i16);
                    self.record.visit_shares.push((f64::from(self.tree.visits(edge)) / f64::from(visits)) as f32);
                }
                None => {
                    self.record.visit_moves.push(settings.no_move);
                    self.record.visit_shares.push(0.0);
                }
            }
        }
        encode_board(&self.tree.board, &mut self.record.boards);
        //the root keeps its score for the side that moved into it, the opponent of the side to move here
        self.record.root_scores.push((1.0 - root.mean_score()) as f32);
        self.record.white_to_move.push(u8::from(turn == Color::White));

        let chosen = if self.plies < settings.sampled_plies { self.sample_by_visits() } else { self.tree.most_visited_edge(0) };
        self.tree.play(chosen, settings.tree_reuse);
        self.plies += 1;
        self.noised = false;
        self.add_noise_once_expanded(settings);
    }

    //a root's edge drawn with the chance of its share of the visits (random.choice(p=visits / visits.sum()))
    fn sample_by_visits(&mut self) -> usize {
        let edges = self.tree.root().edges();
        let total: u32 = edges.clone().map(|edge| self.tree.visits(edge)).sum();
        let mut draw = self.random.random_range(0..total);
        for edge in edges {
            if draw < self.tree.visits(edge) {
                return edge;
            }
            draw -= self.tree.visits(edge);
        }
        unreachable!("the draw is below the total")
    }
}
