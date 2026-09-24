import json

import pytest

from tools.play import elo_log
from tools.play.elo_log import elo_margin, lineage, log_result, read_results


#a run fine-tuned from a run fine-tuned from a run trained from scratch is round 2 of the first one's lineage
def test_the_round_is_counted_along_the_init_checkpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(elo_log, "ARTIFACTS_DIR", tmp_path / "artifacts")
    runs = tmp_path / "artifacts" / "runs"
    for run, init in (("start", None), ("round1", "artifacts/runs/start/model.pt"), ("round2", "artifacts/runs/round1/model.pt")):
        (runs / run).mkdir(parents=True)
        (runs / run / "settings.json").write_text(json.dumps({"init_checkpoint": init}))

    assert lineage(runs / "round2" / "model.pt") == ("start", 2)
    assert lineage(runs / "start" / "model.pt") == ("start", 0)


def test_a_result_is_logged_with_its_elo(tmp_path, monkeypatch):
    monkeypatch.setattr(elo_log, "ELO_RESULTS", tmp_path / "elo_results.jsonl")
    monkeypatch.setattr(elo_log, "ELO_TENSORBOARD", tmp_path / "elo")
    (tmp_path / "start").mkdir()

    even = log_result(tmp_path / "start" / "model.pt", "mcts", 64, "stockfish 1700", 1700, 90, 90, 20)
    one_sided = log_result(tmp_path / "start" / "model.pt", "argmax", 64, "stockfish 1320", 1320, 199, 0, 1)

    assert even["elo"] == pytest.approx(1700) and even["play"] == "mcts 64" and 40 < even["margin"] < 60
    assert one_sided["elo"] is None and one_sided["play"] == "policy"
    assert len(read_results()) == 2
    assert list((tmp_path / "elo").iterdir()) #the tensorboard chart was written


#the interval shrinks with more games of the same score
def test_the_margin_shrinks_with_more_games():
    assert elo_margin(400, 400, 200) < elo_margin(40, 40, 20)
