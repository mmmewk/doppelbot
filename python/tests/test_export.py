"""Parity checks for the explicit-embedding Maia-3 ONNX export wrapper."""

import numpy as np
import onnxruntime as ort
import torch

from doppelbot.export import Maia3EmbeddingWrapper, export_artifacts, interpolate_elo
from doppelbot.scorers import Maia3Scorer


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