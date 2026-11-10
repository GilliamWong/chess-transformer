#wdl (.rtbw) is the result under perfect play, dtz (.rtbz) the plies to the next capture or pawn move
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import re
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
import chess
import chess.syzygy
from tqdm import tqdm

SYZYGY_DIR = Path(__file__).parent.parent / "artifacts" / "engines" / "syzygy"
LICHESS_TABLES = "https://tablebase.lichess.ovh/tables/standard/"
#ranked_moves needs the dtz tables of this many pieces
TABLEBASE_PIECES = 5
CONNECTIONS = 2
CHUNK_BYTES = 1 << 20
TRIES = 5
RETRY_SECONDS = 30
TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class TableSet:
    pieces: int
    folder: Path
    lichess_name: str
    tables_per_half: int

    def count(self, extension: str) -> int:
        return len(list(self.folder.glob(f"*.{extension}")))

#a size's tables probe the smaller sizes' too (the positions after a capture)
TABLE_SETS = (TableSet(5, SYZYGY_DIR, "3-4-5", 145), TableSet(6, SYZYGY_DIR.with_name("syzygy6"), "6", 365),
              TableSet(7, SYZYGY_DIR.with_name("syzygy7"), "7", 1001))
MOST_PIECES = max(table_set.pieces for table_set in TABLE_SETS)

def table_sets(pieces: int) -> list[TableSet]:
    if pieces not in {table_set.pieces for table_set in TABLE_SETS}:
        raise ValueError(f"there are tables of up to {', '.join(str(table_set.pieces) for table_set in TABLE_SETS)} pieces, not {pieces}")
    return [table_set for table_set in TABLE_SETS if table_set.pieces <= pieces]

def tablebase_path(pieces: int) -> str:
    return os.pathsep.join(str(table_set.folder) for table_set in table_sets(pieces))


#into name.part, renamed once whole so that an unfinished download is not taken for a table
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

#appends what the .part lacks; raises unless it then has `size` bytes (lichess drops some connections)
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

#wdl first, as the search needs only those; run again to resume
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


#above TABLEBASE_PIECES the dtz tables may be left out whole: those positions are then searched, not ranked
def usable(wdl: int, dtz: int, whole: int, pieces: int) -> bool:
    return wdl == whole and (dtz == whole or dtz == 0 and pieces > TABLEBASE_PIECES)

def open_tablebase(pieces: int = TABLEBASE_PIECES) -> chess.syzygy.Tablebase:
    tablebase = chess.syzygy.Tablebase()
    for table_set in table_sets(pieces):
        wdl, dtz, whole = table_set.count("rtbw"), table_set.count("rtbz"), table_set.tables_per_half
        if not usable(wdl, dtz, whole, table_set.pieces):
            raise FileNotFoundError(f"{table_set.folder} has {wdl} of the {whole} wdl and {dtz} of the {whole} dtz tables, "
                                    f"python -m engine.tablebase --pieces {table_set.pieces} downloads them")
        tablebase.add_directory(str(table_set.folder))
    return tablebase

#the folders of a SyzygyPath, whatever tables they hold as in stockfish and lc0: a missing table's positions are searched
def open_tablebase_path(path: str) -> chess.syzygy.Tablebase:
    tablebase = chess.syzygy.Tablebase()
    for folder in path_folders(path):
        tablebase.add_directory(folder)
    return tablebase

#the most pieces of a wdl table in `path`, up to `limit`; 0 if none
def most_pieces(path: str, limit: int) -> int:
    return min(max((len(name) - 1 for name in found_tables(path_folders(path))["rtbw"]), default=0), limit)

def path_folders(path: str) -> list[str]:
    return [folder for folder in path.split(os.pathsep) if folder]

#a link to nothing (a drive not mounted) is no table
def found_tables(folders: list[str]) -> dict[str, set[str]]:
    found = {"rtbw": set(), "rtbz": set()}
    for folder in folders:
        for name in os.listdir(folder):
            table, _, extension = name.partition(".")
            if extension in found and chess.syzygy.is_tablename(table) and os.path.isfile(os.path.join(folder, name)):
                found[extension].add(table)
    return found

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

#best first; ties keep the order of the legal moves. None if the probes are not done by `deadline` (time.perf_counter's): tables on a slow disk; or a table is missing
def ranked_moves(tablebase: chess.syzygy.Tablebase, board: chess.Board, deadline: float | None = None) -> list[TablebaseMove] | None:
    moves = []
    clock = board.halfmove_clock
    for move in list(board.legal_moves):
        if deadline is not None and time.perf_counter() >= deadline:
            return None
        zeroing = board.is_zeroing(move)
        board.push(move)
        try:
            #the tables answer for the opponent, hence the minus
            if board.is_checkmate():
                wdl, plies = 2, 0
            elif zeroing:
                wdl, plies = -tablebase.probe_wdl(board), 1
            else:
                wdl, plies = -tablebase.probe_wdl(board), abs(tablebase.probe_dtz(board)) + 1
                #the tables count the 50-move rule from a clock of 0 at the next position: the root's clock and this move's plies must fit in 100
                if abs(wdl) == 2 and clock + plies > 100:
                    wdl //= 2
        except chess.syzygy.MissingTableError:
            return None
        finally:
            board.pop()
        moves.append(TablebaseMove(move, wdl, plies))

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
