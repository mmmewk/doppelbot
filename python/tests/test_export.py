"""Parity checks for the explicit-embedding Maia-3 ONNX export wrapper."""

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from doppelbot.export import PRECISIONS, Maia3EmbeddingWrapper, export_artifacts, export_variant, interpolate_elo
from doppelbot.games import Game
from doppelbot.positions import user_positions
from doppelbot.precision import Outputs, compare, position_inputs, run_onnx
from doppelbot.scorers import Maia3Scorer

# A short game with castling, captures, a black-to-move position and an en-passant capture.
MOVES = "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6 f1e2 e7e5 d4b3 f8e7 e1g1 e8g8 f2f4 b7b5 f4f5 b5b4 c3d5 f6e4 a2a4 b4a3".split()
GAME = Game(
    site="test", id="g1", played_at=0, speed="rapid", rated=True, base=600, inc=0,
    white="w", black="b", white_elo=1250, black_elo=1430, moves=MOVES, clocks=[],
)
# Top-1 agreement, mean KL and the largest probability difference allowed against fp32 on those positions.
TOLERANCE = {"fp16": (1.0, 1e-4, 0.02), "int8": (0.9, 1e-2, 0.15)}


def test_embedding_wrapper_matches_maia3_forward():
    scorer = Maia3Scorer("maia3-5m", device="cpu")
    model = scorer.model
    tokens = torch.rand(2, 64, 97)
    self_elos = torch.tensor([800.0, 2100.0])
    oppo_elos = torch.tensor([1300.0, 1750.0])

    with torch.no_grad():
        expected = model(tokens, self_elos, oppo_elos)
        actual = Maia3EmbeddingWrapper(model)(
            tokens,
            interpolate_elo(model, self_elos),
            interpolate_elo(model, oppo_elos),
        )

    for expected_output, actual_output in zip(expected, actual):
        torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=1e-6)


def test_exported_onnx_matches_wrapper(tmp_path):
    onnx_path = export_artifacts("maia3-5m", tmp_path)
    scorer = Maia3Scorer("maia3-5m", device="cpu")
    model = scorer.model
    wrapper = Maia3EmbeddingWrapper(model)
    tokens = torch.rand(1, 64, 97)
    self_emb = interpolate_elo(model, torch.tensor([1150.0]))
    oppo_emb = interpolate_elo(model, torch.tensor([1600.0]))

    with torch.no_grad():
        expected = wrapper(tokens, self_emb, oppo_emb)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual = session.run(
        None,
        {
            "tokens": tokens.numpy(),
            "self_emb": self_emb.detach().numpy(),
            "oppo_emb": oppo_emb.detach().numpy(),
        },
    )

    for expected_output, actual_output in zip(expected, actual):
        np.testing.assert_allclose(actual_output, expected_output.numpy(), rtol=1e-4, atol=1e-5)


@pytest.fixture(scope="module")
def fp32_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("artifacts") / "maia3-5m"
    export_artifacts("maia3-5m", directory)
    return directory


@pytest.mark.parametrize("precision", PRECISIONS)
def test_variant_matches_fp32(fp32_dir, precision):
    variant = export_variant(fp32_dir, precision)
    assert variant.parent.name == f"maia3-5m-{precision}"
    for name in ("moves.json", "elo_emb.json"):
        assert (variant.parent / name).read_bytes() == (fp32_dir / name).read_bytes()
    assert variant.stat().st_size < 0.6 * (fp32_dir / "model.onnx").stat().st_size

    # Inputs and outputs stay fp32 whatever the weights are, so the TypeScript runtime needs no changes.
    graph = onnx.load(variant).graph
    for value in [*graph.input, *graph.output]:
        assert value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT

    scorer = Maia3Scorer("maia3-5m", real_history=True, device="cpu")
    inputs = position_inputs(scorer, user_positions([GAME], "w") + user_positions([GAME], "b"))
    reference = run_onnx(fp32_dir / "model.onnx", inputs)
    metrics = compare(reference, run_onnx(variant, inputs), inputs.target)
    top1, kl, prob_diff = TOLERANCE[precision]
    assert metrics["top1_agreement"] >= top1
    assert metrics["mean_kl"] < kl
    assert metrics["max_abs_prob_diff"] < prob_diff


def test_compare_identical_outputs_is_exact():
    probabilities = np.array([[0.7, 0.3, 0.0], [0.2, 0.5, 0.3]])
    outputs = Outputs(probabilities, np.array([[0.2, 0.3, 0.5]] * 2))
    metrics = compare(outputs, outputs, np.array([0, 2]))
    assert metrics["top1_agreement"] == 1.0
    assert metrics["mean_kl"] == 0.0
    assert metrics["max_abs_prob_diff"] == 0.0
    assert metrics["fp32_logp"] == metrics["variant_logp"] == pytest.approx((np.log(0.7) + np.log(0.3)) / 2)
