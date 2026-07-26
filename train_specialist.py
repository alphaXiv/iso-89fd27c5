"""Matched short full-parameter fine-tuning for one public GLUE specialist."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["sst2", "qnli"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    cfg = json.loads(Path("experiment_config.json").read_text())
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained("outputs/base")
    model = AutoModelForSequenceClassification.from_pretrained("outputs/base")
    raw = load_dataset("glue", args.task, split="train")
    n = min(int(cfg["train_examples_per_task"]), len(raw))
    shuffled = raw.shuffle(seed=args.seed).select(range(n))

    if args.task == "sst2":
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

    train_ds = shuffled.map(
        tokenize,
        batched=True,
        remove_columns=[c for c in shuffled.column_names if c != "label"],
    )
    train_args = TrainingArguments(
        output_dir=args.output,
        overwrite_output_dir=True,
        learning_rate=float(cfg["learning_rate"]),
        max_steps=int(cfg["max_steps"]),
        per_device_train_batch_size=int(cfg["train_batch_per_gpu"]),
        warmup_ratio=0.1,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        logging_steps=10,
        save_strategy="no",
        eval_strategy="no",
        bf16=True,
        seed=args.seed,
        data_seed=args.seed,
        report_to="none",
        ddp_find_unused_parameters=False,
    )
    print(
        "TRAIN_CONFIG "
        f"task={args.task} seed={args.seed} examples={n} "
        f"steps={cfg['max_steps']} lr={cfg['learning_rate']} "
        f"world_size={train_args.world_size}",
        flush=True,
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    result = trainer.train()
    trainer.save_model(args.output)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.output)
        print(
            "SPECIALIST_DONE "
            f"task={args.task} seed={args.seed} "
            f"train_loss={result.training_loss:.6f} "
            f"runtime_s={result.metrics.get('train_runtime', float('nan')):.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
