#every finished run side by side, grouped by budget: a table in the terminal and artifacts/progress.html
#   python -m training.compare_runs
import html
import json
from pathlib import Path
from training.data import ARTIFACTS_DIR
from engine.checkpoint import RUNS_DIR

PROGRESS_PAGE = ARTIFACTS_DIR / "progress.html"

COLUMNS = ["budget", "name", "note", "d_model", "layers", "heads", "params", "lr", "positions", "minutes", "val_loss", "val_acc", "commit"]

def load_runs() -> list[dict]:
    runs = []
    for settings_path in RUNS_DIR.glob("*/settings.json"):
        settings = json.loads(settings_path.read_text())
        #a run that was stopped early has its settings but no results
        if "final_val_loss" not in settings:
            continue

        model_settings = settings["model_settings"]
        #the oldest runs recorded only positions_budget
        positions = settings.get("positions_seen", settings.get("positions_budget", 0))
        runs.append({
            "budget": settings.get("budget", f"{round(positions / 1e6):g}M"),
            "name": settings_path.parent.name,
            "note": settings.get("note", ""),
            "d_model": model_settings["d_model"],
            "layers": model_settings["num_layers"],
            "heads": model_settings["n_heads"],
            "params": f"{settings['parameters'] / 1e6:.1f}M",
            "lr": f"{settings['peak_learn_rate']:g}",
            "positions": f"{positions / 1e6:.1f}M",
            "minutes": f"{settings['train_seconds'] / 60:.1f}",
            "val_loss": f"{settings['final_val_loss']:.4f}",
            "val_acc": f"{settings['final_val_accuracy']:.2%}",
            "commit": settings.get("commit", ""),
            "accuracy": settings["final_val_accuracy"],
            "finished": settings.get("finished_at", "") or settings_path.stat().st_mtime_ns,
            "order": settings_path.stat().st_mtime_ns,
        })

    #budget groups together, inside a group the best run first
    return sorted(runs, key=lambda run: (run["budget"], run["val_loss"]))

def print_table(runs: list[dict]) -> None:
    widths = {column: max(len(column), *(len(str(run[column])) for run in runs)) for column in COLUMNS}
    print("  ".join(column.ljust(widths[column]) for column in COLUMNS))
    previous_budget = None
    for run in runs:
        if previous_budget is not None and run["budget"] != previous_budget:
            print()
        previous_budget = run["budget"]
        print("  ".join(str(run[column]).ljust(widths[column]) for column in COLUMNS))

#one chart per budget: every run in the order it finished, and a line through the best accuracy so far
def chart_svg(runs: list[dict]) -> str:
    runs = sorted(runs, key=lambda run: run["order"])
    width, height, left, right, top, bottom = 760, 260, 50, 20, 16, 30
    low = min(run["accuracy"] for run in runs) - 0.01
    high = max(run["accuracy"] for run in runs) + 0.01

    def x(index: int) -> float:
        return left + (width - left - right) * (index + 0.5) / len(runs)

    def y(accuracy: float) -> float:
        return top + (height - top - bottom) * (high - accuracy) / (high - low)

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="val accuracy run after run">']
    for tick in range(5):
        accuracy = low + (high - low) * tick / 4
        parts.append(f'<line class="grid" x1="{left}" x2="{width - right}" y1="{y(accuracy):.1f}" y2="{y(accuracy):.1f}"/>')
        parts.append(f'<text class="axis" x="{left - 6}" y="{y(accuracy) + 4:.1f}" text-anchor="end">{accuracy:.1%}</text>')

    best_so_far, best_points = 0.0, []
    for index, run in enumerate(runs):
        best_so_far = max(best_so_far, run["accuracy"])
        best_points.append(f"{x(index):.1f},{y(best_so_far):.1f}")
    parts.append(f'<polyline class="best" points="{" ".join(best_points)}"/>')

    for index, run in enumerate(runs):
        label = html.escape(f'{run["name"]}: {run["val_acc"]}' + (f' ({run["note"]})' if run["note"] else ""))
        parts.append(f'<circle class="run" cx="{x(index):.1f}" cy="{y(run["accuracy"]):.1f}" r="5"><title>{label}</title></circle>')
        parts.append(f'<text class="axis" x="{x(index):.1f}" y="{height - 10}" text-anchor="middle">{index + 1}</text>')
    parts.append("</svg>")
    return "".join(parts)

def write_progress_page() -> Path:
    runs = load_runs()
    sections = []
    for budget in sorted({run["budget"] for run in runs}):
        group = [run for run in runs if run["budget"] == budget]
        by_time = sorted(group, key=lambda run: run["order"])
        numbers = {run["name"]: index + 1 for index, run in enumerate(by_time)}
        rows = "".join(
            "<tr>" + f"<td>{numbers[run['name']]}</td>" + "".join(f"<td>{html.escape(str(run[column]))}</td>" for column in COLUMNS[1:]) + "</tr>"
            for run in group
        )
        header = "<th>#</th>" + "".join(f"<th>{column}</th>" for column in COLUMNS[1:])
        sections.append(
            f"<section><h2>budget {html.escape(budget)} <small>{len(group)} runs, best {max(run['accuracy'] for run in group):.2%}</small></h2>"
            f"{chart_svg(group)}<p class='hint'>dots are runs in the order they finished (# in the table), the line is the best so far. Hover a dot for its name.</p>"
            f"<div class='scroll'><table><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table></div></section>"
        )

    PROGRESS_PAGE.write_text(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="30"><title>Chess Transformer runs</title>
<style>
:root {{ --bg: #ffffff; --fg: #1c1c1e; --muted: #6b6b70; --line: #d9d9de; --accent: #2f6fde; --dot: #1c1c1e; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg: #16161a; --fg: #ececf0; --muted: #9a9aa2; --line: #34343c; --accent: #7aa7ff; --dot: #ececf0; }} }}
body {{ background: var(--bg); color: var(--fg); font: 15px/1.5 system-ui, sans-serif; margin: 0 auto; max-width: 1100px; padding: 24px 16px; }}
h1 {{ font-size: 22px; margin: 0 0 4px; }} h2 {{ font-size: 17px; margin: 32px 0 8px; }} small, .hint {{ color: var(--muted); font-weight: normal; font-size: 13px; }}
svg {{ width: 100%; height: auto; }} .grid {{ stroke: var(--line); }} .axis {{ fill: var(--muted); font-size: 11px; }}
.best {{ fill: none; stroke: var(--accent); stroke-width: 2; }} .run {{ fill: var(--dot); }}
.scroll {{ overflow-x: auto; }} table {{ border-collapse: collapse; width: 100%; font-size: 13px; white-space: nowrap; }}
th, td {{ text-align: left; padding: 5px 10px; border-bottom: 1px solid var(--line); }} th {{ color: var(--muted); font-weight: 600; }}
tbody tr:first-child td {{ font-weight: 600; }}
</style></head><body>
<h1>Chess Transformer runs</h1>
<p class="hint">Validation accuracy on the full val set. Only runs with the same budget are comparable. Within a budget the best run is first. The page reloads every 30 seconds.</p>
{"".join(sections) if sections else "<p>No finished runs yet.</p>"}
</body></html>
""", encoding="utf-8")
    return PROGRESS_PAGE

def main():
    runs = load_runs()
    if len(runs) == 0:
        print(f"no finished runs with a settings.json in {RUNS_DIR}")
        return

    print_table(runs)
    print(f"\npage: {write_progress_page()}")


if __name__ == "__main__":
    main()
