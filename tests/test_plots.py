import numpy as np

from tools.plot_attention import MIN_SHARE, load_data, named_heads, save_data


#the pictures are redrawn from this file alone, so everything they need has to come back: the weights (to fp16), the
#sink column a model without a sink leaves NaN, what each head looks up, and the position
def test_attention_data_comes_back_from_its_file(tmp_path):
    rng = np.random.default_rng(0)
    blocks = [rng.random((2, 64, 71)).astype(np.float32) for _ in range(3)]
    for block in blocks:
        block[..., 70] = np.nan
    kinds = [[{"kind": "knight", "entropy": 1.5, "what": "knight's move (90%)"}] * 2] * 3
    tokens = rng.integers(0, 13, 70)

    save_data(tmp_path / "attention_data.npz", blocks, kinds, tokens, position=40)
    loaded_blocks, loaded_kinds, loaded_tokens, position = load_data(tmp_path / "attention_data.npz")

    for block, loaded in zip(blocks, loaded_blocks, strict=True):
        assert np.allclose(loaded, block, atol=1e-3, equal_nan=True) and np.isnan(loaded[..., 70]).all()
    assert loaded_kinds == kinds and np.array_equal(loaded_tokens, tokens) and position == 40


def kind(knight=0.0, ray=0.0, offset=0.0, step=(0, 0), whole_board=0.0, meta=0.0, piece="empty") -> dict:
    return {"knight_share": knight, "ray_share": ray, "offset_share": offset, "offset_df": step[0], "offset_dr": step[1],
            "global_share": whole_board, "meta_share": meta, "piece_at_target": piece}


#the clearest head of each kind: the fixed-step heads each a different step, and a kind no head follows well enough
#left out rather than named after a head that does not do it
def test_the_named_heads_are_the_clearest_of_each_kind():
    kinds = [[kind(knight=0.9), kind(knight=0.95, offset=0.5, step=(1, 2)), kind(offset=0.6, step=(0, 1))],
             [kind(offset=0.55, step=(0, 1)), kind(offset=0.45, step=(1, -1)), kind(ray=MIN_SHARE / 2, whole_board=0.7, piece="their king")]]

    names = {name: (block, head) for block, head, name in named_heads(kinds)}

    assert names["block00_head01_knight_attacker.png"] == (0, 1)
    #(0, 1) twice: the stronger one only; the knight head's (1, 2) is a step too
    assert {name for name in names if "_step_" in name} == {"block00_head02_step_+0_+1.png", "block00_head01_step_+1_+2.png", "block01_head01_step_+1_-1.png"}
    assert names["block01_head02_one_square_for_the_board_their_king.png"] == (1, 2)
    assert not any("first_piece_on_line" in name or "metadata" in name for name in names)
