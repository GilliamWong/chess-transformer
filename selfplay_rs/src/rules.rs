//the rules exactly as the python has them: python-chess, encoding.py, mcts.game_result and final_result in
//tests/reference_selfplay.py (tests/test_selfplay_rs.py holds the two together)
use shakmaty::zobrist::Zobrist64;
use shakmaty::{Bitboard, Board, CastlingSide, Chess, Color, EnPassantMode, Move, MoveList, Position, Role, Square};
use shakmaty_syzygy::{SyzygyError, Tablebase, Wdl};

pub const TOKENS_PER_BOARD: usize = 70;
//encoding.py's token offsets
const TURN_TOKEN_OFFSET: u8 = 13;
const CASTLING_TOKEN_OFFSET: u8 = 15;
const EN_PASSANT_TOKEN_OFFSET: u8 = 17;
const PIECE_TYPES: u8 = 6;
//the rules' threefold repetition (mcts.py's search has its own)
const THREEFOLD: usize = 3;

//python-chess's _transposition_key as a zobrist hash: en passant only counts when the capture is legal
pub fn position_key(board: &Chess) -> u64 {
    board.zobrist_hash::<Zobrist64>(EnPassantMode::Legal).0
}

pub fn after(board: &Chess, chess_move: Move) -> Chess {
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
        Move::Castle { king, rook } => {
            let file = if rook > king { 6 } else { 2 };
            (king.to_usize(), king.to_usize() / 8 * 8 + file, 0)
        }
        _ => (
            chess_move.from().expect("no drops in chess").to_usize(),
            chess_move.to().to_usize(),
            chess_move.promotion().map_or(0, |role| role as usize),
        ),
    }
}

//python-chess's generation order, which is mcts.py's children order: of equal children the first is searched and played
pub fn legal_moves_in_python_order(board: &Chess) -> MoveList {
    let mut moves = board.legal_moves();
    let in_check = board.is_check();
    moves.sort_unstable_by_key(|&chess_move| python_order(chess_move, in_check));
    moves
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

//queen, rook, bishop, knight
fn promotion_order(promotion: Option<Role>) -> u32 {
    match promotion {
        None | Some(Role::Queen) => 0,
        Some(Role::Rook) => 1,
        Some(Role::Bishop) => 2,
        Some(_) => 3,
    }
}

//Board.is_insufficient_material
pub fn is_insufficient_material(board: &Board) -> bool {
    has_insufficient_material(board, Color::White) && has_insufficient_material(board, Color::Black)
}

//Board.has_insufficient_material, line for line
fn has_insufficient_material(board: &Board, color: Color) -> bool {
    let ours = board.by_color(color);
    if (ours & (board.pawns() | board.rooks() | board.queens())).any() {
        return false;
    }
    if (ours & board.knights()).any() {
        return ours.count() <= 2 && (board.by_color(!color) & !board.kings() & !board.queens()).is_empty();
    }
    if (ours & board.bishops()).any() {
        let same_color = (board.bishops() & Bitboard::DARK_SQUARES).is_empty() || (board.bishops() & Bitboard::LIGHT_SQUARES).is_empty();
        return same_color && board.pawns().is_empty() && board.knights().is_empty();
    }
    true
}

//occurrences of the last key (positions oldest first), counting back to the last capture or pawn move
pub fn repetitions(keys: &[u64], halfmoves: u32) -> usize {
    let (current, earlier) = keys.split_last().expect("the position itself");
    let since_zeroing = &earlier[earlier.len() - earlier.len().min(halfmoves as usize)..];
    1 + since_zeroing.iter().filter(|&key| key == current).count()
}

//the syzygy tables (tools.tablebase)
pub struct Tables {
    tablebase: Tablebase<Chess>,
    pieces: usize,
}

impl Tables {
    pub fn open(folder: &str, pieces: usize) -> std::io::Result<Tables> {
        let mut tablebase = Tablebase::new();
        tablebase.add_directory(folder)?;
        Ok(Tables { tablebase, pieces })
    }

    //tools.tablebase.covers
    fn covers(&self, board: &Chess) -> bool {
        board.board().occupied().count() <= self.pieces && board.castles().castling_rights().is_empty()
    }

    //tools.tablebase.wdl_score: a win or loss the fifty move rule turns into a draw is a draw
    fn score(&self, board: &Chess) -> Result<f64, SyzygyError> {
        Ok(match self.tablebase.probe_wdl_after_zeroing(board)? {
            Wdl::Win => 1.0,
            Wdl::Loss => 0.0,
            _ => 0.5,
        })
    }
}

//mcts.game_result; keys end with the board's own
pub fn game_result(
    board: &Chess,
    keys: &[u64],
    legal_moves: &MoveList,
    draw: f64,
    repetitions_for_a_draw: usize,
    tables: Option<&Tables>,
) -> Result<Option<f64>, SyzygyError> {
    if legal_moves.is_empty() {
        return Ok(Some(if board.is_check() { 0.0 } else { draw }));
    }
    if is_insufficient_material(board.board()) || board.halfmoves() >= 100 || repetitions(keys, board.halfmoves()) >= repetitions_for_a_draw
    {
        return Ok(Some(draw));
    }
    if let Some(tables) = tables
        && tables.covers(board)
    {
        return Ok(Some(tables.score(board)?));
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
    if is_insufficient_material(board.board())
        || fifty_moves
        || repetitions(keys, halfmoves) >= THREEFOLD
        || threefold_after_a_move(board, keys, &legal_moves)
    {
        return Ok(Some(0.5));
    }
    if let Some(tables) = tables
        && tables.covers(board)
    {
        return Ok(Some(white_score(tables.score(board)?)));
    }
    if plies >= max_plies {
        return Ok(Some(0.5));
    }
    Ok(None)
}

//a legal move reaches a position for the third time (the second half of can_claim_threefold_repetition)
fn threefold_after_a_move(board: &Chess, keys: &[u64], legal_moves: &MoveList) -> bool {
    let since_zeroing = &keys[keys.len() - 1 - (keys.len() - 1).min(board.halfmoves() as usize)..];
    legal_moves.iter().any(|&chess_move| {
        let key = position_key(&after(board, chess_move));
        since_zeroing.iter().filter(|&&earlier| earlier == key).count() >= THREEFOLD - 1
    })
}
