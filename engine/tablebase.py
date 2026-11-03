#syzygy endgame tables for positions without castling rights: the result under perfect play (wdl, .rtbw) and the plies
#to the next capture or pawn move (dtz, .rtbz). python -m engine.tablebase downloads those of up to 5 pieces into
#artifacts/engines/syzygy, about 1 GB; --pieces 6 adds the 6 piece tables in artifacts/engines/syzygy6, 73 GB of wdl
#and 87 GB of dtz (lichess sends ~3 MB/s a connection: ~7 h over 2), or with --wdl-only their wdl half alone. The 7
#piece tables (~17 TB, a tournament box's) are not downloaded here: a SyzygyPath or artifacts/engines/syzygy7 holds them
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import re
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
import chess
import chess.syzygy
from tqdm import tqdm

#not from data.ARTIFACTS_DIR: data imports torch, and labelling workers import this file
SYZYGY_DIR = Path(__file__).parent.parent / "artifacts" / "engines" / "syzygy"
LICHESS_TABLES = "https://tablebase.lichess.ovh/tables/standard/"
#the tables used unless more pieces are asked for; ranked_moves needs their dtz tables
TABLEBASE_PIECES = 5
CONNECTIONS = 2
CHUNK_BYTES = 1 << 20
#a table's download is tried this many times, this long apart; a connection that sends nothing this long has failed
TRIES = 5
RETRY_SECONDS = 30
TIMEOUT_SECONDS = 60


#the tables in one pair of lichess folders, <lichess_name>-wdl and <lichess_name>-dtz, and the folder they go into
@dataclass(frozen=True)
class TableSet:
    pieces: int #the most pieces of the positions they cover
    folder: Path
    lichess_name: str
    tables_per_half: int

    #the tables of one half in the folder, "rtbw" (wdl) or "rtbz" (dtz)
    def count(self, extension: str) -> int:
        return len(list(self.folder.glob(f"*.{extension}")))

#a size's tables probe the smaller sizes' too (the positions after a capture). Lichess keeps the 7 piece tables in
#subfolders by material, not as one -wdl and -dtz pair
TABLE_SETS = (TableSet(5, SYZYGY_DIR, "3-4-5", 145), TableSet(6, SYZYGY_DIR.with_name("syzygy6"), "6", 365),
              TableSet(7, SYZYGY_DIR.with_name("syzygy7"), "7", 1001))
#the most pieces of any set, and SyzygyProbeLimit's default: a SyzygyPath gives the most it holds up to it
MOST_PIECES = max(table_set.pieces for table_set in TABLE_SETS)

#the sets that cover positions of up to `pieces` pieces
def table_sets(pieces: int) -> list[TableSet]:
    if pieces not in {table_set.pieces for table_set in TABLE_SETS}:
        raise ValueError(f"there are tables of up to {', '.join(str(table_set.pieces) for table_set in TABLE_SETS)} pieces, not {pieces}")
    return [table_set for table_set in TABLE_SETS if table_set.pieces <= pieces]

#the folders of the tables up to `pieces`, as a SyzygyPath (os.pathsep between them), for rules.rs Tables
def tablebase_path(pieces: int) -> str:
    return os.pathsep.join(str(table_set.folder) for table_set in table_sets(pieces))


#one table into its .part file, which is renamed once whole so that an unfinished download is not taken for a table. A
#connection that fails or ends short (lichess drops some) is tried again from where it stopped, TRIES times in all.
#Stops between chunks once stopping is set
def download_table(folder: Path, url: str, name: str, size: int, progress: tqdm, stopping: threading.Event) -> None:
    path = folder / name
    partial = path.with_name(name + ".part")
    if path.exists():
        progress.update(size)
        return
    progress.update(min(partial.stat().st_size, size) if partial.exists() else 0)
    for attempt in range(1, TRIES + 1):
        try:
            download_rest(url + name, partial, size, progress, stopping)
            break
        except OSError:
            if attempt == TRIES or stopping.wait(RETRY_SECONDS):
                raise
    if not stopping.is_set():
        partial.replace(path)

#the bytes the .part does not have yet, appended to it; raises unless it then has the table's size
def download_rest(url: str, partial: Path, size: int, progress: tqdm, stopping: threading.Event) -> None:
    have = partial.stat().st_size if partial.exists() else 0
    if have > size:
        partial.unlink()
        have = 0
    if have < size:
        request = urllib.request.Request(url, headers={"Range": f"bytes={have}-"})
        #206: the bytes after those the .part has; 200: the whole table, from a server that ignores the range
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response, open(partial, "ab" if response.status == 206 else "wb") as table_file:
            while chunk := response.read(CHUNK_BYTES):
                if stopping.is_set():
                    return
                table_file.write(chunk)
                progress.update(len(chunk))
    if partial.stat().st_size != size:
        raise IOError(f"{partial.name} has {partial.stat().st_size:,} of {size:,} bytes")

#the set's tables that are not in its folder yet, the wdl half first (the search needs only those), each checked against
#the size lichess lists. Run again to resume
def download(table_set: TableSet, connections: int = CONNECTIONS, dtz: bool = True) -> None:
    if table_set.pieces == 7:
        raise ValueError(f"the 7 piece tables are ~17 TB: put them in {table_set.folder} or give the engine a SyzygyPath")
    table_set.folder.mkdir(parents=True, exist_ok=True)
    tables = [] #(url, name, bytes)
    for half in ("wdl", "dtz") if dtz else ("wdl",):
        url = f"{LICHESS_TABLES}{table_set.lichess_name}-{half}/"
        with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response:
            listed = dict(re.findall(r'href="([KQRBNP]+v[KQRBNP]+\.rtb[wz])">[^<]*</a>\s+\S+\s+\S+\s+(\d+)', response.read().decode()))
        if len(listed) != table_set.tables_per_half:
            raise ValueError(f"{url} lists {len(listed)} tables, {table_set.tables_per_half} expected")
        tables += [(url, name, int(size)) for name, size in sorted(listed.items())]

    stopping = threading.Event()
    with tqdm(total=sum(size for *_, size in tables), unit="B", unit_scale=True, desc=table_set.folder.name, dynamic_ncols=True) as progress:
        pool = ThreadPoolExecutor(connections)
        try:
            #in order, so the first failure among them is raised
            for downloading in [pool.submit(download_table, table_set.folder, *table, progress, stopping) for table in tables]:
                downloading.result()
        finally:
            #after a failure or ctrl+c the tables not begun are left, and those begun stop at their next chunk
            stopping.set()
            pool.shutdown(cancel_futures=True)


#tables of up to `pieces` pieces, `whole` of each half, can be opened: all the wdl tables are there, and all the dtz
#tables or, above TABLEBASE_PIECES, none
def usable(wdl: int, dtz: int, whole: int, pieces: int) -> bool:
    return wdl == whole and (dtz == whole or dtz == 0 and pieces > TABLEBASE_PIECES)

#the tables of positions of up to `pieces` pieces. Each size needs all its wdl tables, and up to TABLEBASE_PIECES all
#its dtz tables; a bigger size's dtz tables may be left out as a whole (the 6 piece wdl tables fit a pod's 80 GB disk,
#with their dtz tables they don't), and ranked_pieces tells which sizes ranked_moves can rank
def open_tablebase(pieces: int = TABLEBASE_PIECES) -> chess.syzygy.Tablebase:
    tablebase = chess.syzygy.Tablebase()
    for table_set in table_sets(pieces):
        wdl, dtz, whole = table_set.count("rtbw"), table_set.count("rtbz"), table_set.tables_per_half
        if not usable(wdl, dtz, whole, table_set.pieces):
            raise FileNotFoundError(f"{table_set.folder} has {wdl} of the {whole} wdl and {dtz} of the {whole} dtz tables, "
                                    f"python -m engine.tablebase --pieces {table_set.pieces} downloads them")
        tablebase.add_directory(str(table_set.folder))
    return tablebase

#the tables in the folders of a SyzygyPath (os.pathsep between them, as uci's SyzygyPath option gives them), each size
#up to `pieces` checked by the tables' names as open_tablebase checks a downloaded set; and the most pieces of a
#position ranked_moves can rank with them
def open_tablebase_path(path: str, pieces: int) -> tuple[chess.syzygy.Tablebase, int]:
    folders = path_folders(path)
    found = found_tables(folders)
    ranked = 0
    for size in range(3, pieces + 1):
        wdl, dtz, whole = size_counts(found, size)
        if not usable(wdl, dtz, whole, size):
            raise FileNotFoundError(f"{path} has {wdl} of the {whole} wdl and {dtz} of the {whole} dtz tables of {size} pieces")
        ranked = size if dtz else ranked
    tablebase = chess.syzygy.Tablebase()
    for folder in folders:
        tablebase.add_directory(folder)
    return tablebase, ranked

#the most pieces, up to `limit`, whose tables the folders of `path` hold as open_tablebase_path asks, as stockfish takes
#the tables it finds up to its SyzygyProbeLimit; 2 if not even the 3 piece ones are all there
def complete_pieces(path: str, limit: int) -> int:
    found = found_tables(path_folders(path))
    pieces = 2
    while pieces < limit and usable(*size_counts(found, pieces + 1), pieces + 1):
        pieces += 1
    return pieces

def path_folders(path: str) -> list[str]:
    return [folder for folder in path.split(os.pathsep) if folder]

#the tables' names in the folders, by half ("rtbw", "rtbz")
def found_tables(folders: list[str]) -> dict[str, set[str]]:
    return {extension: {name[:-len(extension) - 1] for folder in folders for name in os.listdir(folder) if name.endswith("." + extension)}
            for extension in ("rtbw", "rtbz")}

#(wdl, dtz, every table) of the `size` piece tables among those found
def size_counts(found: dict[str, set[str]], size: int) -> tuple[int, int, int]:
    names = {name for name in chess.syzygy.tablenames(piece_count=size) if len(name) == size + 1}
    return len(names & found["rtbw"]), len(names & found["rtbz"]), len(names)

#the most pieces of a position ranked_moves can rank with open_tablebase(pieces): those of the sizes whose dtz tables are
#there (open_tablebase makes sure of those up to TABLEBASE_PIECES)
def ranked_pieces(pieces: int) -> int:
    return max(table_set.pieces for table_set in table_sets(pieces) if table_set.count("rtbz") > 0)

#syzygy leaves out positions with castling rights
def covers(board: chess.Board, pieces: int = TABLEBASE_PIECES) -> bool:
    return chess.popcount(board.occupied) <= pieces and not board.castling_rights

#a win or loss the fifty move rule turns into a draw (wdl 1, -1) is a draw
def wdl_score(wdl: int) -> float:
    return 1.0 if wdl == 2 else 0.0 if wdl == -2 else 0.5


#a legal move as the tables see it, for the side playing it
@dataclass
class TablebaseMove:
    move: chess.Move
    wdl: int
    plies_to_zeroing: int #with perfect play: 1 if this move is a capture or pawn move, 0 for a mate

    #smaller is better: the better result, then the fastest win or the longest loss
    def rank(self) -> tuple[int, int]:
        return (-self.wdl, self.plies_to_zeroing if self.wdl > 0 else -self.plies_to_zeroing if self.wdl < 0 else 0)

#best first; ties keep their order, preferred moves first among them
def ranked_moves(tablebase: chess.syzygy.Tablebase, board: chess.Board, preferred: list[chess.Move] = []) -> list[TablebaseMove]:
    legal_moves = list(board.legal_moves)
    ordered = [move for move in preferred if move in legal_moves] + [move for move in legal_moves if move not in preferred]

    moves = []
    for move in ordered:
        zeroing = board.is_zeroing(move)
        board.push(move)
        #the tables answer for the opponent, hence the minus
        if board.is_checkmate():
            moves.append(TablebaseMove(move, 2, 0))
        elif zeroing:
            moves.append(TablebaseMove(move, -tablebase.probe_wdl(board), 1))
        else:
            wdl, plies = -tablebase.probe_wdl(board), abs(tablebase.probe_dtz(board)) + 1
            #the probe counts the 50-move rule from the next position, but this move used one ply of it
            if abs(wdl) == 2 and plies > 100:
                wdl //= 2
            moves.append(TablebaseMove(move, wdl, plies))
        board.pop()

    return sorted(moves, key=TablebaseMove.rank)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="download the syzygy tables, resuming where the last run stopped")
    parser.add_argument("--pieces", type=int, choices=[table_set.pieces for table_set in TABLE_SETS], default=TABLEBASE_PIECES,
                        help="the tables of positions of up to this many pieces, default %(default)s")
    parser.add_argument("--wdl-only", action="store_true",
                        help="leave out the dtz tables above 5 pieces (the 6 piece tables in 73 GB, not 160): the engine then searches those positions")
    parser.add_argument("--connections", type=int, default=CONNECTIONS, help="tables downloaded at once, default %(default)s")
    arguments = parser.parse_args()
    for table_set in table_sets(arguments.pieces):
        download(table_set, arguments.connections, dtz=table_set.pieces <= TABLEBASE_PIECES or not arguments.wdl_only)
        print(f"{table_set.count('rtbw')} wdl and {table_set.count('rtbz')} dtz tables in {table_set.folder}")
