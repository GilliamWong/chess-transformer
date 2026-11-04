use shakmaty::uci::UciMove;
use shakmaty::zobrist::Zobrist64;
use shakmaty::{CastlingSide, Chess, Color, EnPassantMode, File, Move, MoveList, Position, Role, Square};
use shakmaty_syzygy::filesystem::MmapFilesystem;
use shakmaty_syzygy::{SyzygyError, Tablebase, Wdl};
use std::sync::Arc;

pub const TOKENS_PER_BOARD: usize = 70;
//encoding.py's token offsets
const TURN_TOKEN_OFFSET: u8 = 13;
const CASTLING_TOKEN_OFFSET: u8 = 15;
const EN_PASSANT_TOKEN_OFFSET: u8 = 17;
const PIECE_TYPES: u8 = 6;
const THREEFOLD: usize = 3;

//python-chess's _transposition_key as a zobrist hash: en passant only counts when the capture is legal
pub fn position_key(board: &Chess) -> u64 {
    board.zobrist_hash::<Zobrist64>(EnPassantMode::Legal).0
}

fn after(board: &Chess, chess_move: Move) -> Chess {
    let mut next = board.clone();
    next.play_unchecked(chess_move);
    next
}

//encoding.encode_board: from the side to move, black's boards mirrored (square ^ 56)
pub fn encode_board(board: &Chess, tokens: &mut Vec<u8>) {
    let turn = board.turn();
    let flip = if turn == Color::White { 0 } else { 56 };
    let start = tokens.len();
    tokens.resize(start + TOKENS_PER_BOARD, 0);
    let encoding = &mut tokens[start..];
    for (square, piece) in board.board() {
        let opponent = if piece.color == turn { 0 } else { PIECE_TYPES };
        encoding[square.to_usize() ^ flip] = piece.role as u8 + opponent;
    }
    let castles = board.castles();
    encoding[64] = TURN_TOKEN_OFFSET + u8::from(turn == Color::White);
    encoding[65] = CASTLING_TOKEN_OFFSET + u8::from(castles.has(turn, CastlingSide::KingSide));
    encoding[66] = CASTLING_TOKEN_OFFSET + u8::from(castles.has(turn, CastlingSide::QueenSide));
    encoding[67] = CASTLING_TOKEN_OFFSET + u8::from(castles.has(!turn, CastlingSide::KingSide));
    encoding[68] = CASTLING_TOKEN_OFFSET + u8::from(castles.has(!turn, CastlingSide::QueenSide));
    encoding[69] = EN_PASSANT_TOKEN_OFFSET + board.ep_square(EnPassantMode::Legal).map_or(0, |square| (square.to_usize() ^ flip) as u8);
}

//selfplay.move_id_table: 64 from squares x 64 to squares x 7 (no promotion, then python-chess's piece types), -1 for none
pub struct MoveIds(Vec<i16>);

pub const MOVE_ID_TABLE_LENGTH: usize = 64 * 64 * 7;

impl MoveIds {
    pub fn new(table: Vec<i16>) -> MoveIds {
        assert_eq!(table.len(), MOVE_ID_TABLE_LENGTH);
        MoveIds(table)
    }

    //encoding.encode_move_to_id for the side playing the move
    pub fn id(&self, chess_move: Move, turn: Color) -> usize {
        let flip = if turn == Color::White { 0 } else { 56 };
        let (from, to, promotion) = python_chess_squares(chess_move);
        let move_id = self.0[((from ^ flip) * 64 + (to ^ flip)) * 7 + promotion];
        usize::try_from(move_id).unwrap_or_else(|_| panic!("{chess_move:?} has no move id"))
    }
}

//a move's squares and promotion the way python-chess writes the move: castling as the king's two steps
fn python_chess_squares(chess_move: Move) -> (usize, usize, usize) {
    match chess_move {
        Move::Castle { king, rook } => (king.to_usize(), castle_target(king, rook).to_usize(), 0),
        _ => (
            chess_move.from().expect("no drops in chess").to_usize(),
            chess_move.to().to_usize(),
            chess_move.promotion().map_or(0, |role| role as usize),
        ),
    }
}

fn castle_target(king: Square, rook: Square) -> Square {
    Square::from_coords(if rook > king { File::G } else { File::C }, king.rank())
}

//a tree edge's move in 2 bytes, read in the position it is played from: from bits 0..6, to 6..12 (a castle's rook), kind 12..16
//(0 plain, 1 to 4 a promotion to knight, bishop, rook or queen, EN_PASSANT, CASTLE)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PackedMove(u16);

const EN_PASSANT: u16 = 5;
const CASTLE: u16 = 6;

impl PackedMove {
    pub fn new(chess_move: Move) -> PackedMove {
        let squares = |from: Square, to: Square| from.to_u32() | to.to_u32() << 6;
        let code = match chess_move {
            Move::Normal { from, to, promotion, .. } => squares(from, to) | promotion.map_or(0, |role| u32::from(role) - 1) << 12,
            Move::EnPassant { from, to } => squares(from, to) | u32::from(EN_PASSANT) << 12,
            Move::Castle { king, rook } => squares(king, rook) | u32::from(CASTLE) << 12,
            Move::Put { .. } => unreachable!("no drops in chess"),
        };
        PackedMove(code as u16)
    }

    pub fn from_bits(bits: u16) -> PackedMove {
        PackedMove(bits)
    }

    pub fn bits(self) -> u16 {
        self.0
    }

    fn from(self) -> Square {
        Square::new(u32::from(self.0 & 63))
    }

    fn to(self) -> Square {
        Square::new(u32::from(self.0 >> 6 & 63))
    }

    fn kind(self) -> u16 {
        self.0 >> 12
    }

    fn promotion(self) -> Option<Role> {
        (1..=4).contains(&self.kind()).then(|| Role::ALL[usize::from(self.kind())])
    }

    //the move in `board`, the position it was packed in
    pub fn to_move(self, board: &Chess) -> Move {
        let (from, to) = (self.from(), self.to());
        match self.kind() {
            EN_PASSANT => Move::EnPassant { from, to },
            CASTLE => Move::Castle { king: from, rook: to },
            _ => Move::Normal {
                role: board.board().role_at(from).expect("the moved piece"),
                from,
                capture: board.board().role_at(to),
                to,
                promotion: self.promotion(),
            },
        }
    }

    //Move::to_uci(CastlingMode::Standard)'s uci, no position needed
    pub fn to_uci(self) -> UciMove {
        let (from, to) = (self.from(), self.to());
        let to = if self.kind() == CASTLE { castle_target(from, to) } else { to };
        UciMove::Normal { from, to, promotion: self.promotion() }
    }
}

//python-chess's generation order, which is mcts.py's children order: of equal children the first is searched and played
pub fn legal_moves_in_python_order(board: &Chess) -> MoveList {
    let mut moves = board.legal_moves();
    sort_moves_in_python_order(board, &mut moves);
    moves
}

pub fn sort_moves_in_python_order(board: &Chess, moves: &mut MoveList) {
    let in_check = board.is_check();
    //key << 9 | index sorted on the stack: sort_by_cached_key's stable order without its allocation on every call
    let mut keys = [0u32; 256];
    let keys = &mut keys[..moves.len()];
    for (index, (key, &chess_move)) in keys.iter_mut().zip(moves.iter()).enumerate() {
        *key = python_order(chess_move, in_check) << 9 | index as u32;
    }
    keys.sort_unstable();
    let unsorted = moves.clone();
    for (chess_move, key) in moves.iter_mut().zip(keys.iter()) {
        *chess_move = unsorted[(key & 511) as usize];
    }
}

fn python_order(chess_move: Move, in_check: bool) -> u32 {
    //python-chess walks its bitboards from the highest square down
    let down = |square: Square| 63 - square.to_u32();
    let (group, first, second, promotion) = match chess_move {
        //in check, the king's moves come first (_generate_evasions)
        Move::Normal { role: Role::King, to, .. } if in_check => (0, 0, down(to), 0),
        Move::Normal { role, from, to, .. } if role != Role::Pawn => (1, down(from), down(to), 0),
        //kingside first
        Move::Castle { rook, .. } => (2, down(rook), 0, 0),
        Move::Normal { from, to, capture: Some(_), promotion, .. } => (3, down(from), down(to), promotion_order(promotion)),
        Move::Normal { from, to, promotion, .. } if from.to_u32().abs_diff(to.to_u32()) == 8 => {
            (4, down(to), 0, promotion_order(promotion))
        }
        Move::Normal { to, .. } => (5, down(to), 0, 0),
        Move::EnPassant { from, .. } => (6, down(from), 0, 0),
        Move::Put { .. } => unreachable!("no drops in chess"),
    };
    group << 16 | first << 10 | second << 4 | promotion
}

fn promotion_order(promotion: Option<Role>) -> u32 {
    match promotion {
        None | Some(Role::Queen) => 0,
        Some(Role::Rook) => 1,
        Some(Role::Bishop) => 2,
        Some(_) => 3,
    }
}

//the keys back to the last capture or pawn move, the last key (the position itself) included
fn since_zeroing(keys: &[u64], halfmoves: u32) -> &[u64] {
    let earlier = keys.len() - 1;
    &keys[earlier - earlier.min(halfmoves as usize)..]
}

//occurrences of the last key, positions oldest first
fn repetitions(keys: &[u64], halfmoves: u32) -> usize {
    let current = keys.last().expect("the position itself");
    since_zeroing(keys, halfmoves).iter().filter(|&key| key == current).count()
}

pub struct Tables {
    tablebase: Tablebase<Chess>,
    pieces: usize,
}

impl Tables {
    //folders as a SyzygyPath (engine.tablebase.tablebase_path): ';' between them on windows, ':' elsewhere
    pub fn open(folders: &str, pieces: usize) -> std::io::Result<Tables> {
        //SAFETY: the files stay unchanged while mapped; a failed page read ends the process rather than erroring
        let mut tablebase = Tablebase::with_filesystem(Arc::new(unsafe { MmapFilesystem::new() }.with_advise_random(true)));
        for folder in std::env::split_paths(folders) {
            tablebase.add_directory(folder)?;
        }
        Ok(Tables { tablebase, pieces })
    }

    pub fn covers(&self, board: &Chess) -> bool {
        board.board().occupied().count() <= self.pieces && board.castles().castling_rights().is_empty()
    }

    //engine.tablebase.wdl_score with a draw scored `draw`: a win or loss the fifty move rule turns into a draw is a draw
    fn score(&self, board: &Chess, draw: f64) -> Result<f64, SyzygyError> {
        Ok(match self.tablebase.probe_wdl_after_zeroing(board)? {
            Wdl::Win => 1.0,
            Wdl::Loss => 0.0,
            _ => draw,
        })
    }
}

//mcts.game_result; keys end with the board's own
pub fn game_result(
    board: &Chess,
    keys: &[u64],
    legal_moves: &MoveList,
    draw: f64,
    tables: Option<&Tables>,
) -> Result<Option<f64>, SyzygyError> {
    if legal_moves.is_empty() {
        return Ok(Some(if board.is_check() { 0.0 } else { draw }));
    }
    if board.is_insufficient_material() || board.halfmoves() >= 100 || repetitions(keys, board.halfmoves()) >= THREEFOLD {
        return Ok(Some(draw));
    }
    if let Some(tables) = tables
        && tables.covers(board)
    {
        return Ok(Some(tables.score(board, draw)?));
    }
    Ok(None)
}

//reference_selfplay.final_result (outcome(claim_draw=True), the tablebase, max_plies); keys end with the board's own
pub fn final_result(
    board: &Chess,
    keys: &[u64],
    plies: usize,
    max_plies: usize,
    tables: Option<&Tables>,
) -> Result<Option<f64>, SyzygyError> {
    let white_score = |score: f64| if board.turn() == Color::White { score } else { 1.0 - score };
    let legal_moves = board.legal_moves();
    if legal_moves.is_empty() {
        return Ok(Some(if board.is_check() { white_score(0.0) } else { 0.5 }));
    }
    let halfmoves = board.halfmoves();
    //the fifty move rule, and a draw the side to move can claim with its move (can_claim_fifty_moves)
    let fifty_moves = halfmoves >= 100
        || (halfmoves >= 99
            && legal_moves.iter().any(|&chess_move| !chess_move.is_zeroing() && !after(board, chess_move).legal_moves().is_empty()));
    if board.is_insufficient_material()
        || fifty_moves
        || repetitions(keys, halfmoves) >= THREEFOLD
        || threefold_after_a_move(board, keys, &legal_moves)
    {
        return Ok(Some(0.5));
    }
    if let Some(tables) = tables
        && tables.covers(board)
    {
        return Ok(Some(white_score(tables.score(board, 0.5)?)));
    }
    if plies >= max_plies {
        return Ok(Some(0.5));
    }
    Ok(None)
}

//a legal move reaches a position for the third time (the second half of can_claim_threefold_repetition)
fn threefold_after_a_move(board: &Chess, keys: &[u64], legal_moves: &MoveList) -> bool {
    let recent = since_zeroing(keys, board.halfmoves());
    legal_moves.iter().any(|&chess_move| {
        let key = position_key(&after(board, chess_move));
        recent.iter().filter(|&&earlier| earlier == key).count() >= THREEFOLD - 1
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use shakmaty::CastlingMode;
    use shakmaty::fen::Fen;

    //positions with every kind of move: castles, en passant, promotions to each piece with and without captures, checks
    const POSITIONS: [&str; 6] = [
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 3",
        "1k6/6P1/8/8/8/8/8/K7 w - - 0 1",
        "r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 0 1",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    ];

    #[test]
    fn a_move_comes_back_from_its_packing() {
        let mut kinds = [0; 7];
        for fen in POSITIONS {
            let board: Chess = fen.parse::<Fen>().expect("a fen").into_position(CastlingMode::Standard).expect("a position");
            let mut packings = std::collections::HashSet::new();
            for chess_move in board.legal_moves() {
                let packed = PackedMove::new(chess_move);
                assert_eq!(packed.to_move(&board), chess_move);
                assert_eq!(packed.to_uci(), chess_move.to_uci(CastlingMode::Standard));
                assert!(packings.insert(packed), "{chess_move:?} shares its packing");
                kinds[usize::from(packed.kind())] += 1;
            }
        }
        assert!(kinds.iter().all(|&kind| kind > 0), "{kinds:?}");
    }
}
