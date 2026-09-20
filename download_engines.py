#downloads the two programs evaluate_elo.py needs into engines/, which is too big for git and gitignored:
#   stockfish   the opponent the model is measured against
#   cute chess  cutechess-cli runs the matches, cutechess is the gui for playing the model yourself
#it also tells the cute chess gui where the model and stockfish are, so they show up in its engine list.
#
#training does not need any of this, only evaluate_elo.py does. Run it once: python download_engines.py
import json
import os
import platform
import sys
import urllib.request
import zipfile
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
ENGINES_DIR = PROJECT_DIR / "engines"

#official release pages: github.com/official-stockfish/Stockfish and github.com/cutechess/cutechess
STOCKFISH_URL = "https://github.com/official-stockfish/Stockfish/releases/download/sf_19/stockfish-windows-x86-64-universal.zip"
CUTECHESS_URL = "https://github.com/cutechess/cutechess/releases/download/v1.5.1/cutechess-1.5.1-win64.zip"

#where each zip puts its program. evaluate_elo.py imports these, so the paths are only written down here
STOCKFISH = ENGINES_DIR / "stockfish" / "stockfish-windows-x86-64-universal.exe"
CUTECHESS_DIR = ENGINES_DIR / "cutechess" / "cutechess-1.5.1-win64"
CUTECHESS_CLI = CUTECHESS_DIR / "cutechess-cli.exe"
CUTECHESS_GUI = CUTECHESS_DIR / "cutechess.exe"

#downloads and unzips into unzip_into, unless expected_file is already there
def download(url: str, unzip_into: Path, expected_file: Path) -> None:
    if expected_file.exists():
        print(f"already have {expected_file.name}")
        return

    print(f"downloading {url}")
    ENGINES_DIR.mkdir(exist_ok=True)
    zip_path = ENGINES_DIR / url.split("/")[-1]
    urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(unzip_into)
    zip_path.unlink()

    if not expected_file.exists():
        raise FileNotFoundError(f"{url} did not contain {expected_file.relative_to(unzip_into)}")

#the gui keeps its engine list in engines.json. Our two entries are added or updated by name,
#any engines that are already configured there are left alone
def register_with_gui() -> None:
    config_file = Path(os.environ["APPDATA"]) / "cutechess" / "engines.json"

    our_engines = [
        {
            "name": "ChessTransformer",
            "command": f'"{sys.executable}" uci.py',
            "workingDirectory": str(PROJECT_DIR),
            "protocol": "uci"
        },
        {
            "name": "Stockfish 1320",
            "command": f'"{STOCKFISH}"',
            "workingDirectory": str(STOCKFISH.parent),
            "protocol": "uci",
            "options": [{"name": "UCI_LimitStrength", "value": True}, {"name": "UCI_Elo", "value": 1320}]
        }
    ]
    our_names = [engine["name"] for engine in our_engines]

    existing_engines = json.loads(config_file.read_text()) if config_file.exists() else []
    other_engines = [engine for engine in existing_engines if engine.get("name") not in our_names]

    config_file.parent.mkdir(exist_ok=True)
    config_file.write_text(json.dumps(other_engines + our_engines, indent=4))
    print(f"registered {', '.join(our_names)} in {config_file}")

def main():
    if platform.system() != "Windows" or platform.machine() not in ("AMD64", "x86_64"):
        raise SystemExit(
            "this script only knows the 64 bit windows downloads. On another system install stockfish and "
            "cute chess yourself and point STOCKFISH, CUTECHESS_CLI and CUTECHESS_GUI in this file at them"
        )

    download(STOCKFISH_URL, ENGINES_DIR, STOCKFISH)
    download(CUTECHESS_URL, ENGINES_DIR / "cutechess", CUTECHESS_CLI)
    register_with_gui()

    print("done, evaluate_elo.py is ready once main.py has trained a model")


if __name__ == "__main__":
    main()
