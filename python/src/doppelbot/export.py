"""Export Maia-3 with explicit player-vector inputs for browser inference."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import warnings
from pathlib import Path

import onnx
import torch
import torch.nn as nn
from maia3.uci import Maia3UCIEngine, parse_args
from maia3.utils import get_all_possible_moves
from onnxconverter_common import float16
from onnxruntime.quantization import QuantType, quantize_dynamic

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"
PRECISIONS = ("fp16", "int8")


class ExportableRMSNorm(nn.Module):
    """Equivalent to torch.nn.RMSNorm using operators supported by ONNX opset 18."""

    def __init__(self, norm: nn.RMSNorm):
        super().__init__()
        self.weight = norm.weight
        self.eps = norm.eps or torch.finfo(norm.weight.dtype).eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.square().mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


def _make_onnx_exportable(module: nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.RMSNorm):
            setattr(module, name, ExportableRMSNorm(child))
        else:
            _make_onnx_exportable(child)


class Maia3EmbeddingWrapper(torch.nn.Module):
    """The Maia-3 network after Elo interpolation, with room for future LoRA inputs."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.history = model.cfg.history
        self.include_time_info = model.cfg.include_time_info
        self.head_hid_dim = model.cfg.head_hid_dim

    def forward(
        self,
        tokens: torch.Tensor,
        self_emb: torch.Tensor,
        oppo_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        token_dims = 12 * self.history + (3 if self.include_time_info else 0)
        tokens = tokens[:, :, :token_dims]
        self_emb = self_emb.unsqueeze(1).expand(-1, 64, -1)
        oppo_emb = oppo_emb.unsqueeze(1).expand(-1, 64, -1)
        x = self.model.token_projection(torch.cat([tokens, self_emb, oppo_emb], dim=-1))

        if hasattr(self.model, "abs_pe"):
            x = self.model.abs_pe(x)
        x = self.model.transformer(x)

        sq_from = self.model.proj_sq_from(x[:, :64, :])
        sq_to = self.model.proj_sq_to(x[:, :64, :])
        scores_base = torch.einsum("bid,bjd->bij", sq_from, sq_to) / math.sqrt(self.head_hid_dim)
        scores_flat = scores_base.reshape(x.size(0), 64 * 64)

        rank7_indices = list(range(48, 56))
        rank8_indices = list(range(56, 64))
        rank8_features = sq_to[:, rank8_indices, :]
        promo_biases = self.model.promo_bias_proj(rank8_features) * math.sqrt(self.head_hid_dim)
        promotion_base = scores_base[:, rank7_indices][:, :, rank8_indices]
        promotion_logits = (promotion_base.unsqueeze(-1) + promo_biases.unsqueeze(1)).reshape(-1, 256)
        logits_move = torch.cat([scores_flat, promotion_logits], dim=1)

        x = self.model.last_ln(x.mean(dim=1))
        logits_value = self.model.fc_value(torch.relu(self.model.fc_value_hid(x)))
        logits_ponder = self.model.fc_ponder(torch.relu(self.model.fc_ponder_hid(x))).squeeze(1)
        return logits_move, logits_value, logits_ponder


def interpolate_elo(model: torch.nn.Module, elos: torch.Tensor) -> torch.Tensor:
    """Match Maia-3's rating-to-vector interpolation for export validation."""
    return model.interpolate_elo(elos)


def export_artifacts(model_name: str, output_dir: Path, batch_size: int = 2) -> Path:
    """Export an ONNX model and the static data needed by the TypeScript runtime."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = parse_args(["--model", model_name, "--device", "cpu"])
    engine = Maia3UCIEngine(cfg)
    engine.ensure_model_loaded()
    model = engine.model.eval()
    wrapper = Maia3EmbeddingWrapper(model).eval()
    _make_onnx_exportable(wrapper)

    tokens = torch.zeros(batch_size, 64, 12 * cfg.history + 1, dtype=torch.float32)
    self_emb = interpolate_elo(model, torch.full((batch_size,), 1500.0))
    oppo_emb = interpolate_elo(model, torch.full((batch_size,), 1500.0))
    onnx_path = output_dir / "model.onnx"
    torch.onnx.export(
        wrapper,
        (tokens, self_emb, oppo_emb),
        onnx_path,
        input_names=["tokens", "self_emb", "oppo_emb"],
        output_names=["logits_move", "logits_value", "logits_ponder"],
        dynamic_axes={
            "tokens": {0: "batch"},
            "self_emb": {0: "batch"},
            "oppo_emb": {0: "batch"},
            "logits_move": {0: "batch"},
            "logits_value": {0: "batch"},
            "logits_ponder": {0: "batch"},
        },
        opset_version=18,
        dynamo=False,
    )
    (output_dir / "moves.json").write_text(json.dumps(get_all_possible_moves()) + "\n")
    (output_dir / "elo_emb.json").write_text(
        json.dumps(
            {
                "low": model.elo_embedding_low.weight.detach().cpu().flatten().tolist(),
                "high": model.elo_embedding_high.weight.detach().cpu().flatten().tolist(),
            }
        )
        + "\n"
    )
    return onnx_path


def convert_fp16(source: Path, destination: Path) -> None:
    """Store weights as fp16. The inputs and outputs stay fp32, so the TypeScript runtime is unchanged."""
    model = onnx.load(source)
    # The converter retypes every float tensor but leaves Cast(to=float32) alone, which breaks the attention
    # scale computed from the tensor shape (Shape -> Cast -> Sqrt). Retarget those casts to fp16 first.
    for node in model.graph.node:
        if node.op_type == "Cast":
            for attribute in node.attribute:
                if attribute.name == "to" and attribute.i == onnx.TensorProto.FLOAT:
                    attribute.i = onnx.TensorProto.FLOAT16
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # tiny constants (eps) round to the smallest fp16 value
        converted = float16.convert_float_to_float16(model, keep_io_types=True, check_fp16_ready=False)
    onnx.save(converted, destination)


def quantize_int8(source: Path, destination: Path) -> None:
    """Dynamic quantization: int8 weights, activations quantized at run time."""
    quantize_dynamic(source, destination, weight_type=QuantType.QInt8)


def export_variant(source_dir: Path, precision: str) -> Path:
    """Write <source_dir>-<precision>/ next to an fp32 export, with identical moves.json and elo_emb.json."""
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    output_dir = source_dir.with_name(f"{source_dir.name}-{precision}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("moves.json", "elo_emb.json"):
        shutil.copyfile(source_dir / name, output_dir / name)
    onnx_path = output_dir / "model.onnx"
    convert = convert_fp16 if precision == "fp16" else quantize_int8
    convert(source_dir / "model.onnx", onnx_path)
    return onnx_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Maia-3 for browser inference")
    parser.add_argument("--model", default="maia3-5m")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--precision",
        choices=("fp32", *PRECISIONS, "all"),
        default="fp32",
        help="fp16 and int8 derive from the fp32 export in --output-dir; all exports fp32 first",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or ARTIFACTS_DIR / args.model
    if args.precision in ("fp32", "all"):
        print(export_artifacts(args.model, output_dir))
    for precision in PRECISIONS if args.precision == "all" else (args.precision,):
        if precision != "fp32":
            print(export_variant(output_dir, precision))


if __name__ == "__main__":
    main()