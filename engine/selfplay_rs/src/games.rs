//tests/reference_selfplay.py's games move for move, apart from the random numbers
use crate::rules::{MoveIds, Tables, encode_board, final_result};
use crate::search::{SearchSettings, Tree};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use shakmaty::{Chess, Color, Position};
use shakmaty_syzygy::SyzygyError;

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

//every position before a move, with what the search found there
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

    pub fn awaits_noise(&self) -> bool {
        !self.noised
    }

    pub fn add_noise_once_expanded(&mut self, settings: &GameSettings) {
        if !self.noised && self.tree.root().has_edges() {
            self.tree.add_root_noise(&mut self.random, settings.root_noise_alpha, settings.root_noise_share);
            self.noised = true;
        }
    }

    pub fn play_move(&mut self, settings: &GameSettings, move_ids: &MoveIds) {
        let turn = self.tree.board.turn();
        let root = self.tree.root_index();
        let visits = |edge| self.tree.visits(root, edge);
        let most_visited: Vec<usize> =
            self.tree.edges_by_visits().into_iter().take(settings.top_moves).filter(|&edge| visits(edge) > 0).collect();
        let total: u32 = most_visited.iter().map(|&edge| visits(edge)).sum();
        for slot in 0..settings.top_moves {
            let (id, share) = match most_visited.get(slot) {
                Some(&edge) => {
                    let id = move_ids.id(self.tree.chess_move(edge).to_move(&self.tree.board), turn) as i16;
                    (id, (f64::from(visits(edge)) / f64::from(total)) as f32)
                }
                None => (settings.no_move, 0.0),
            };
            self.record.visit_moves.push(id);
            self.record.visit_shares.push(share);
        }
        encode_board(&self.tree.board, &mut self.record.boards);
        //the root's score is for the side that moved into it, the opponent of the side to move
        self.record.root_scores.push((1.0 - self.tree.root().mean_score()) as f32);
        self.record.white_to_move.push(u8::from(turn == Color::White));

        let chosen = if self.plies < settings.sampled_plies { self.sample_by_visits() } else { self.tree.most_visited_edge(root) };
        self.tree.play(chosen, settings.tree_reuse);
        self.plies += 1;
        self.noised = false;
        self.add_noise_once_expanded(settings);
    }

    fn sample_by_visits(&mut self) -> usize {
        let (root, edges) = (self.tree.root_index(), self.tree.root().edges());
        let total: u32 = edges.clone().map(|edge| self.tree.visits(root, edge)).sum();
        let mut draw = self.random.random_range(0..total);
        for edge in edges {
            let visits = self.tree.visits(root, edge);
            if draw < visits {
                return edge;
            }
            draw -= visits;
        }
        unreachable!("the draw is below the total")
    }
}
