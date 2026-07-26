"""Merge two public-task specialists and print the complete evidence record."""

from __future__ import annotations

import copy
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from iso_merger import polar_columns, project_tangent, thin_svd


RIDGE = 1e-12
CLIP = (0.0, 1.5)


def frame_coordinates(
    w0: torch.Tensor,
    experts: list[torch.Tensor],
    keep_ratio: float,
) -> dict[str, Any]:
    """Released ISO construction, exposed so restoration/mask can be ablated."""
    w0 = w0.to(dtype=torch.float64)
    wes = [w.to(device=w0.device, dtype=torch.float64) for w in experts]
    u0, s0, vh0 = thin_svd(w0)
    v0 = vh0.T.contiguous()
    q = s0.numel()
    keep = max(1, int(round(keep_ratio * q)))
    xis_u, xis_v, spectra = [], [], []
    for we in wes:
        ue, se, vhe = thin_svd(we)
        signs = torch.where(
            (u0 * ue).sum(dim=0) >= 0,
            torch.ones_like(se),
            -torch.ones_like(se),
        )
        ue = ue * signs.unsqueeze(0)
        ve = (vhe * signs.unsqueeze(1)).T.contiguous()
        xu = project_tangent(u0, ue - u0)
        xv = project_tangent(v0, ve - v0)
        if keep < q:
            xu = xu.clone()
            xv = xv.clone()
            xu[:, keep:] = 0
            xv[:, keep:] = 0
        xis_u.append(xu)
        xis_v.append(xv)
        spectra.append(se)

    gs = []
    for xu, xv in zip(xis_u, xis_v):
        gs.append(
            (xu * s0.unsqueeze(0)) @ v0.T
            + (u0 * s0.unsqueeze(0)) @ xv.T
        )
    gram = torch.empty((len(gs), len(gs)), dtype=torch.float64, device=w0.device)
    for i, gi in enumerate(gs):
        for j, gj in enumerate(gs):
            gram[i, j] = (gi * gj).sum()
    gram_np = gram.cpu().numpy()
    diag = np.diag(gram_np)
    active = diag > 1e-24
    coeff = np.zeros(len(gs), dtype=np.float64)
    if active.any():
        ga = gram_np[np.ix_(active, active)]
        ba = np.diag(ga)
        try:
            coeff[active] = np.linalg.solve(
                ga + RIDGE * np.eye(ga.shape[0]), ba
            )
        except np.linalg.LinAlgError:
            coeff[active] = 1.0
    coeff = np.clip(coeff, *CLIP)
    ct = torch.as_tensor(coeff, dtype=torch.float64, device=w0.device)
    xu = sum(ct[i] * xis_u[i] for i in range(len(xis_u)))
    xv = sum(ct[i] * xis_v[i] for i in range(len(xis_v)))
    xu = project_tangent(u0, xu)
    xv = project_tangent(v0, xv)
    us = polar_columns(u0 + xu)
    vs = polar_columns(v0 + xv)
    masked_energy = sum(float((x.square().sum()).item()) for x in xis_u + xis_v)
    return {
        "u": us,
        "v": vs,
        "base_s": s0,
        "expert_mean_s": torch.stack(spectra).mean(dim=0),
        "coeff": coeff,
        "masked_energy": masked_energy,
        "q": q,
        "keep": keep,
    }


def reconstruct(frames: dict[str, Any], spectrum: str) -> torch.Tensor:
    s = frames["base_s"] if spectrum == "base" else frames["expert_mean_s"]
    return (frames["u"] * s.unsqueeze(0)) @ frames["v"].T


def spectral_error(w: torch.Tensor, reference_s: torch.Tensor) -> dict[str, float]:
    observed = torch.linalg.svdvals(w.to(torch.float64))
    delta = observed - reference_s.to(observed)
    scale = max(float(reference_s.max().item()), 1e-30)
    return {
        "max_abs": float(delta.abs().max().item()),
        "max_rel_to_sigma_max": float(delta.abs().max().item() / scale),
        "rmse_rel_to_sigma_max": float(
            torch.sqrt(torch.mean(delta.square())).item() / scale
        ),
    }


def merged_states(
    base: dict[str, torch.Tensor],
    expert_a: dict[str, torch.Tensor],
    expert_b: dict[str, torch.Tensor],
    keep_ratio: float,
    device: torch.device,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    names = ["iso", "iso_no_restore", "iso_no_mask", "task_arithmetic", "average"]
    out = {name: {} for name in names}
    per_matrix: list[dict[str, Any]] = []
    coefficients: list[list[float]] = []
    t0 = time.time()
    for idx, key in enumerate(base):
        w0, wa, wb = base[key], expert_a[key], expert_b[key]
        if w0.is_floating_point() and w0.ndim == 2:
            w0d = w0.to(device)
            experts = [wa.to(device), wb.to(device)]
            masked = frame_coordinates(w0d, experts, keep_ratio)
            unmasked = frame_coordinates(w0d, experts, 1.0)
            iso64 = reconstruct(masked, "base")
            no_restore64 = reconstruct(masked, "expert_mean")
            no_mask64 = reconstruct(unmasked, "base")
            exact = spectral_error(iso64, masked["base_s"])
            saved = spectral_error(iso64.float(), masked["base_s"])
            no_mask_saved = spectral_error(no_mask64.float(), masked["base_s"])
            no_restore_saved = spectral_error(no_restore64.float(), masked["base_s"])
            per_matrix.append(
                {
                    "name": key,
                    "shape": list(w0.shape),
                    "kept_modes": masked["keep"],
                    "total_modes": masked["q"],
                    "iso_float64": exact,
                    "iso_float32_checkpoint": saved,
                    "no_mask_float32_checkpoint": no_mask_saved,
                    "no_restore_vs_base": no_restore_saved,
                    "masked_tangent_energy": masked["masked_energy"],
                    "unmasked_tangent_energy": unmasked["masked_energy"],
                }
            )
            coefficients.append(masked["coeff"].tolist())
            out["iso"][key] = iso64.float().cpu()
            out["iso_no_restore"][key] = no_restore64.float().cpu()
            out["iso_no_mask"][key] = no_mask64.float().cpu()
            del w0d, experts, iso64, no_restore64, no_mask64
        elif w0.is_floating_point():
            mean = w0.float() + 0.5 * ((wa.float() - w0.float()) + (wb.float() - w0.float()))
            for name in ("iso", "iso_no_restore", "iso_no_mask", "average"):
                out[name][key] = mean.to(w0.dtype)
        else:
            for name in ("iso", "iso_no_restore", "iso_no_mask", "average"):
                out[name][key] = w0.clone()

        if w0.is_floating_point():
            out["task_arithmetic"][key] = (
                w0.float() + (wa.float() - w0.float()) + (wb.float() - w0.float())
            ).to(w0.dtype)
            if key not in out["average"]:
                out["average"][key] = (0.5 * (wa.float() + wb.float())).to(w0.dtype)
        else:
            out["task_arithmetic"][key] = w0.clone()
        if idx % 10 == 0:
            print(f"MERGE_PROGRESS tensors={idx + 1}/{len(base)}", flush=True)

    summary = {
        "merge_wall_seconds": time.time() - t0,
        "matrix_count": len(per_matrix),
        "per_matrix": per_matrix,
        "coefficient_mean": np.mean(coefficients, axis=0).tolist(),
        "coefficient_min": np.min(coefficients, axis=0).tolist(),
        "coefficient_max": np.max(coefficients, axis=0).tolist(),
    }
    return out, summary


def make_eval_loader(task: str, tokenizer, cfg: dict[str, Any]) -> DataLoader:
    ds = load_dataset("glue", task, split="validation")
    if task == "sst2":
        def tokenize(batch):
            return tokenizer(
                batch["sentence"],
                truncation=True,
                max_length=int(cfg["max_length"]),
            )
    else:
        def tokenize(batch):
            return tokenizer(
                batch["question"],
                batch["sentence"],
                truncation=True,
                max_length=int(cfg["max_length"]),
            )
    tokenized = ds.map(
        tokenize,
        batched=True,
        remove_columns=[c for c in ds.column_names if c != "label"],
    )
    tokenized.set_format("torch")
    return DataLoader(
        tokenized,
        batch_size=int(cfg["eval_batch"]),
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer),
    )


@torch.inference_mode()
def accuracy(model, loader: DataLoader, device: torch.device) -> float:
    model.to(device).eval()
    correct, count = 0, 0
    for batch in loader:
        labels = batch.pop("labels").to(device)
        inputs = {k: v.to(device) for k, v in batch.items()}
        pred = model(**inputs).logits.argmax(dim=-1)
        correct += int((pred == labels).sum().item())
        count += labels.numel()
    model.to("cpu")
    torch.cuda.empty_cache()
    return correct / count


def main() -> None:
    started = time.time()
    cfg = json.loads(Path("experiment_config.json").read_text())
    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained("outputs/base")
    base_model = AutoModelForSequenceClassification.from_pretrained("outputs/base")
    sst_model = AutoModelForSequenceClassification.from_pretrained("outputs/specialist_sst2")
    qnli_model = AutoModelForSequenceClassification.from_pretrained("outputs/specialist_qnli")
    base_sd = {k: v.detach().cpu() for k, v in base_model.state_dict().items()}
    sst_sd = {k: v.detach().cpu() for k, v in sst_model.state_dict().items()}
    qnli_sd = {k: v.detach().cpu() for k, v in qnli_model.state_dict().items()}
    merged, diagnostics = merged_states(
        base_sd,
        sst_sd,
        qnli_sd,
        float(cfg["iso_keep_ratio"]),
        device,
    )
    loaders = {
        "sst2": make_eval_loader("sst2", tokenizer, cfg),
        "qnli": make_eval_loader("qnli", tokenizer, cfg),
    }
    states = {
        "base": base_sd,
        "specialist_sst2": sst_sd,
        "specialist_qnli": qnli_sd,
        **merged,
    }
    scores: dict[str, dict[str, float]] = {}
    template = copy.deepcopy(base_model)
    for method, state in states.items():
        template.load_state_dict(state, strict=True)
        scores[method] = {
            task: accuracy(template, loader, device)
            for task, loader in loaders.items()
        }
        scores[method]["mixed_balanced"] = 0.5 * (
            scores[method]["sst2"] + scores[method]["qnli"]
        )
        print(
            "EVAL "
            f"method={method} sst2={scores[method]['sst2']:.6f} "
            f"qnli={scores[method]['qnli']:.6f} "
            f"mixed={scores[method]['mixed_balanced']:.6f}",
            flush=True,
        )

    retention: dict[str, dict[str, float | None]] = {}
    specialist_for = {"sst2": "specialist_sst2", "qnli": "specialist_qnli"}
    for method in ("iso", "iso_no_restore", "iso_no_mask", "task_arithmetic", "average"):
        retention[method] = {}
        for task, specialist in specialist_for.items():
            gain = scores[specialist][task] - scores["base"][task]
            retention[method][task] = (
                (scores[method][task] - scores["base"][task]) / gain
                if gain > 1e-12
                else None
            )
        vals = [v for v in retention[method].values() if v is not None]
        retention[method]["mean"] = float(np.mean(vals)) if vals else None

    matrix_rows = diagnostics["per_matrix"]
    exact_max = max(r["iso_float64"]["max_rel_to_sigma_max"] for r in matrix_rows)
    checkpoint_max = max(
        r["iso_float32_checkpoint"]["max_rel_to_sigma_max"] for r in matrix_rows
    )
    no_restore_median = float(
        np.median([r["no_restore_vs_base"]["max_rel_to_sigma_max"] for r in matrix_rows])
    )
    result = {
        "attempt": "clean-recovery-after-2026-07-26T21:36:24.092Z",
        "paper_id": "2607.19331",
        "model_substitution": cfg["model"],
        "task_substitutions": cfg["tasks"],
        "config": cfg,
        "scores": scores,
        "gain_retention": retention,
        "spectrum": {
            "matrix_count": diagnostics["matrix_count"],
            "max_relative_error_float64": exact_max,
            "max_relative_error_float32_checkpoint": checkpoint_max,
            "tolerance_float64": 1e-10,
            "tolerance_float32_checkpoint": 1e-5,
            "all_float64_within_tolerance": exact_max <= 1e-10,
            "all_float32_within_tolerance": checkpoint_max <= 1e-5,
            "no_restore_median_base_spectrum_drift": no_restore_median,
        },
        "iso_diagnostics": {
            k: v for k, v in diagnostics.items() if k != "per_matrix"
        },
        "per_matrix_spectrum": matrix_rows,
        "compute": {
            "backend": "kubernetes",
            "requested_gpu_count": 4,
            "visible_gpu_count": torch.cuda.device_count(),
            "gpu_model": torch.cuda.get_device_name(0),
            "cuda": torch.version.cuda,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "elapsed_wall_seconds": time.time() - started,
        },
    }
    Path("outputs/result.json").write_text(json.dumps(result, indent=2))
    print("RESULT_JSON_BEGIN", flush=True)
    print(json.dumps(result, sort_keys=True), flush=True)
    print("RESULT_JSON_END", flush=True)


if __name__ == "__main__":
    main()
