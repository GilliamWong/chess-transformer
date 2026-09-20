import chess
import pytest
import torch

from tools.checkpoint import load_checkpoint, save_checkpoint
from model import ChessTransformer
from tools.play.player import choose_move, load_model

CPU = torch.device("cpu")

#not the default size, so loading only works if the settings really come from the file
SETTINGS = {"d_model": 48, "num_layers": 3, "n_heads": 6, "dropout": 0.0}


def test_model_of_any_size_comes_back_with_the_same_weights(tmp_path):
    torch.manual_seed(0)
    model = ChessTransformer(**SETTINGS).eval()
    boards = torch.randint(0, 13, (4, 70))
    save_checkpoint(model, SETTINGS, tmp_path / "model.pt")

    loaded, loaded_settings = load_checkpoint(tmp_path / "model.pt", CPU)

    assert loaded_settings == SETTINGS
    assert torch.equal(loaded.eval()(boards), model(boards))


def test_player_loads_a_model_that_is_not_the_default_size(tmp_path):
    save_checkpoint(ChessTransformer(**SETTINGS), SETTINGS, tmp_path / "model.pt")

    model = load_model(tmp_path / "model.pt", CPU)

    assert not model.training
    assert choose_move(model, chess.Board(), CPU) in chess.Board().legal_moves


def test_bare_weights_are_rejected_with_an_explanation(tmp_path):
    torch.save(ChessTransformer().state_dict(), tmp_path / "old.pt")

    with pytest.raises(ValueError, match="model settings"):
        load_checkpoint(tmp_path / "old.pt", CPU)
