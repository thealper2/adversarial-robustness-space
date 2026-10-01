"""Checks for the input-validation and serialisation guarantees the Space relies on."""

import numpy as np
import pytest

from robustlab import artifacts, experiments  # noqa: F401
from robustlab.core import REGISTRY, Param
from robustlab.results import BarChart, ImageGrid, Result, load, save


def test_float_param_is_clamped_and_snapped():
    p = Param.float("eps", "eps", 0.1, 0.0, 0.5, 0.05)
    assert p.coerce(10) == 0.5
    assert p.coerce(-3) == 0.0
    assert p.coerce(0.12) == pytest.approx(0.1)
    assert p.coerce("nan") == 0.1
    assert p.coerce("__import__('os')") == 0.1


def test_int_param_rejects_non_numbers():
    p = Param.int("n", "n", 16, 4, 64, 4)
    assert p.coerce(1e9) == 64
    assert p.coerce(None) == 16
    assert isinstance(p.coerce(10.4), int)


def test_choice_param_only_allows_listed_values():
    p = Param.choice("m", "m", "a", ["a", "b"])
    assert p.coerce("b") == "b"
    assert p.coerce("c") == "a"
    assert p.coerce(["a"]) == "a"


@pytest.mark.parametrize("exp_id", sorted(REGISTRY))
def test_defaults_are_inside_their_domain(exp_id):
    exp = REGISTRY[exp_id]
    assert exp.coerce(exp.defaults()) == exp.defaults()
    assert exp.coerce({"unexpected": 1}) == exp.defaults()


@pytest.mark.parametrize(
    "bad",
    [
        "../secrets",
        "/etc/passwd",
        "models/../../x",
        "other/file.bin",
        "models",
        "models/a b",
    ],
)
def test_artifact_paths_are_confined(bad):
    with pytest.raises(ValueError):
        artifacts.path(bad)


def test_result_roundtrip_without_pickle(tmp_path):
    res = (
        Result()
        .metric("acc", 0.5)
        .add(
            ImageGrid(
                "g",
                np.random.rand(2, 4, 4, 1).astype(np.float32),
                ["a", "b"],
                ["ok", "bad"],
            ),
            BarChart("b", ["x"], {"s": [0.1]}),
        )
    )
    save(res, tmp_path / "r.json", tmp_path / "r.npz")
    with np.load(tmp_path / "r.npz", allow_pickle=False) as npz:
        assert all(npz[k].dtype != object for k in npz.files)
    back = load(tmp_path / "r.json", tmp_path / "r.npz")
    assert back.metrics[0].value == 0.5
    assert back.blocks[0].images.shape == (2, 4, 4, 1)
    assert back.blocks[1].series == {"s": [0.1]}


def test_unknown_block_type_is_rejected(tmp_path):
    (tmp_path / "r.json").write_text(
        '{"schema": 1, "metrics": [], "blocks": [{"type": "Evil"}], "meta": {}}'
    )
    np.savez(tmp_path / "r.npz")
    with pytest.raises(KeyError):
        load(tmp_path / "r.json", tmp_path / "r.npz")
