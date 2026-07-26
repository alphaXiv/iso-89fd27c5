"""Create one exactly shared public base checkpoint, including its task head."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main() -> None:
    cfg = json.loads(Path("experiment_config.json").read_text())
    out = Path("outputs/base")
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20260726)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model"],
        num_labels=2,
        ignore_mismatched_sizes=True,
    )
    model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    print(
        f"BASE_READY model={cfg['model']} parameters="
        f"{sum(p.numel() for p in model.parameters())} path={out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
