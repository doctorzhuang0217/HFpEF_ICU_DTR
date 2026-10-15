#!/usr/bin/env python3
"""
55d_freeze_encoder_versions — Create encoder version registry.

Purpose
-------
Build a JSON registry of all frozen encoder checkpoints with their
metadata, SHA256 hashes, and training metrics. This registry is the
canonical reference for all downstream Step B/C models to ensure
reproducibility and traceability.

Protocol reference: Step 55d (v3.1 § 11)

Inputs
------
- models/50a_encoder_baseline/encoder_baseline_best.pt
- models/55a_encoder_dann/encoder_dann_best.pt  (if available)
- reports/50a_encoder_baseline_summary.txt
- reports/55a_encoder_dann_summary.txt  (if available)

Outputs
-------
- configs/encoder_registry_v1.json
- reports/55d_encoder_registry_summary.txt

Compute: C1
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, resolve_path


def sha256_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_checkpoint_metadata(ckpt_path: Path) -> dict | None:
    """Load checkpoint and extract metadata without loading full model."""
    if not ckpt_path.exists():
        return None

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {})

    meta = {
        "checkpoint_path": str(ckpt_path),
        "sha256": sha256_file(ckpt_path),
        "file_size_mb": round(ckpt_path.stat().st_size / 1e6, 1),
        "epoch": ckpt.get("epoch"),
        "val_loss": round(ckpt.get("val_loss", float("nan")), 6),
        "latent_dim": model_cfg.get("latent_dim"),
        "hidden_dim": model_cfg.get("hidden_dim"),
        "n_layers": model_cfg.get("n_layers"),
        "d_temporal": ckpt.get("d_temporal"),
        "d_static": ckpt.get("d_static"),
        "n_temporal_cols": len(ckpt.get("temporal_cols", [])),
        "n_static_cols": len(ckpt.get("static_cols", [])),
        "frozen": True,
    }

    # Count parameters
    state_dict = ckpt.get("model_state_dict", {})
    n_params = sum(v.numel() for v in state_dict.values())
    meta["n_parameters"] = n_params

    return meta


def main() -> None:
    proj_cfg = load_config()
    project_root = Path(proj_cfg.get("project_root", "."))
    configs_dir = resolve_path(proj_cfg, "configs")
    reports_dir = resolve_path(proj_cfg, "reports")
    configs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("55d_freeze_encoder_versions — Build encoder registry")
    print("=" * 72)

    registry = {
        "version": "v1",
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": "v3.1",
        "encoders": {},
    }

    # ── Baseline encoder (50a) ──────────────────────────────────────────
    print("\n[1/2] Registering baseline encoder (50a) ...")
    baseline_path = project_root / "models/50a_encoder_baseline/encoder_baseline_best.pt"
    baseline_meta = extract_checkpoint_metadata(baseline_path)
    if baseline_meta:
        baseline_meta["encoder_id"] = "baseline_mimic_v1"
        baseline_meta["encoder_type"] = "BiGRU"
        baseline_meta["pretext_tasks"] = ["masked_reconstruction", "next_state_prediction"]
        baseline_meta["domain_adaptation"] = None
        baseline_meta["suitable_for"] = [
            "phenotyping",
            "step_c_raw_input",
            "probe_evaluation",
        ]
        registry["encoders"]["baseline_mimic_v1"] = baseline_meta
        print(f"  ✓ baseline: epoch {baseline_meta['epoch']}, "
              f"val_loss={baseline_meta['val_loss']}, "
              f"params={baseline_meta['n_parameters']:,}")
    else:
        print("  ✗ baseline checkpoint not found")

    # ── DANN encoder (55a) ──────────────────────────────────────────────
    print("\n[2/2] Registering DANN encoder (55a) ...")
    dann_path = project_root / "models/55a_encoder_dann/encoder_dann_best.pt"
    dann_meta = extract_checkpoint_metadata(dann_path)
    if dann_meta:
        ckpt = torch.load(dann_path, map_location="cpu", weights_only=False)
        dann_cfg = ckpt.get("config", {}).get("dann", {})

        dann_meta["encoder_id"] = "dann_mimic_v1"
        dann_meta["encoder_type"] = "BiGRU + DANN"
        dann_meta["pretext_tasks"] = ["masked_reconstruction", "next_state_prediction"]
        dann_meta["domain_adaptation"] = {
            "method": "gradient_reversal",
            "n_domains": ckpt.get("n_domains"),
            "domain_label_strategy": ckpt.get("domain_label_strategy"),
            "domain_loss_weight": dann_cfg.get("domain_loss_weight"),
            "val_domain_loss": round(ckpt.get("val_domain_loss", float("nan")), 6),
        }
        dann_meta["suitable_for"] = [
            "phenotyping",
            "step_c_domain_invariant_input",
            "external_validation",
        ]
        registry["encoders"]["dann_mimic_v1"] = dann_meta
        print(f"  ✓ DANN: epoch {dann_meta['epoch']}, "
              f"val_loss={dann_meta['val_loss']}, "
              f"params={dann_meta['n_parameters']:,}")
    else:
        print("  ✗ DANN checkpoint not found (55a not yet trained)")
        registry["encoders"]["dann_mimic_v1"] = {
            "encoder_id": "dann_mimic_v1",
            "status": "pending",
            "note": "55a training not yet completed",
        }

    # ── Save registry ───────────────────────────────────────────────────
    registry_path = configs_dir / "encoder_registry_v1.json"
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2, default=str)
    print(f"\n  Saved: {registry_path}")

    # ── Summary report ──────────────────────────────────────────────────
    report_lines = [
        "=" * 72,
        "55d — Encoder Version Registry",
        "=" * 72,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Registry: {registry_path}",
        "",
        "REGISTERED ENCODERS",
        "───────────────────",
    ]

    for enc_id, meta in registry["encoders"].items():
        if meta.get("status") == "pending":
            report_lines.append(f"  {enc_id}: PENDING ({meta.get('note', '')})")
            continue
        report_lines.extend([
            f"  {enc_id}:",
            f"    Type: {meta.get('encoder_type', 'unknown')}",
            f"    Epoch: {meta.get('epoch')}",
            f"    Val loss: {meta.get('val_loss')}",
            f"    Params: {meta.get('n_parameters', 0):,}",
            f"    Latent dim: {meta.get('latent_dim')}",
            f"    SHA256: {meta.get('sha256', '')[:16]}...",
            f"    Suitable for: {', '.join(meta.get('suitable_for', []))}",
        ])
        if meta.get("domain_adaptation"):
            da = meta["domain_adaptation"]
            report_lines.append(
                f"    Domain: {da.get('method')} (n={da.get('n_domains')}, "
                f"strategy={da.get('domain_label_strategy')})"
            )

    report_lines.extend([
        "",
        "USAGE",
        "─────",
        "  Step B phenotyping: use pooled latents from either encoder",
        "  Step C primary (75a, 80c): use baseline latents (50c)",
        "  Step C external validation: use DANN latents (55c)",
        "  85c ablation: compare raw vs baseline vs DANN",
        "",
        "  To load an encoder programmatically:",
        "    import json",
        "    reg = json.load(open('configs/encoder_registry_v1.json'))",
        "    ckpt_path = reg['encoders']['baseline_mimic_v1']['checkpoint_path']",
    ])

    report = "\n".join(report_lines) + "\n"
    report_path = reports_dir / "55d_encoder_registry_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Saved: {report_path}")
    print(report)
    print("[55d] Done.")


if __name__ == "__main__":
    main()
