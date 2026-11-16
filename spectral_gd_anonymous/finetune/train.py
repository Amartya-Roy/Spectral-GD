"""
Fine-tuning language models with Spectral-GD vs AdamW over 5 epochs and 3 seeds:
1. GSM8K mathematical-reasoning benchmark with Qwen-3 1.7B
2. kto-mix-14k preference-alignment benchmark with Qwen-1.5 0.5B (Spectral-GD uses lr/5)

Datasets are read from local paths under ./data/gsm8k and ./data/kto-mix-14k.
Logs test loss and test accuracy (%) at the end of each epoch (Epochs 1..5).
"""
import os
import sys
import json
import math
import argparse
from contextlib import nullcontext

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from optimizers import SpectralGD


DEFAULT_MODELS = {
    "gsm8k": "Qwen/Qwen3-1.7B",
    "kto-mix-14k": "Qwen/Qwen1.5-0.5B",
}


class LocalJsonlDataset(Dataset):
    """Loads local JSONL fine-tuning datasets from ./data/<task>/<split>.jsonl."""
    def __init__(self, path: str, tokenizer, max_length: int = 512):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Local dataset file not found at {path}. "
                f"Please place train.jsonl and test.jsonl in the corresponding ./data/ subfolder."
            )
        self.samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                prompt = record.get("question") or record.get("prompt") or ""
                target = record.get("answer") or record.get("completion") or record.get("response") or ""
                full_text = f"{prompt}\n{target}{tokenizer.eos_token or ''}"

                enc = tokenizer(
                    full_text,
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                    return_tensors="pt",
                )
                prompt_ids = tokenizer(prompt, truncation=True, max_length=max_length)["input_ids"]
                input_ids = enc["input_ids"].squeeze(0)
                attention_mask = enc["attention_mask"].squeeze(0)
                labels = input_ids.clone()
                labels[: len(prompt_ids)] = -100
                labels[attention_mask == 0] = -100
                self.samples.append(
                    {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": labels,
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tuning with Spectral-GD vs AdamW (5 epochs, 3 seeds)")
    p.add_argument("--task", type=str, choices=["gsm8k", "kto-mix-14k"], required=True)
    p.add_argument("--optimizer", type=str, choices=["adamw", "spectral_gd"], required=True)
    p.add_argument("--data_dir", type=str, default="", help="Defaults to ./data/<task>")
    p.add_argument("--model_name_or_path", type=str, default="", help="Local path or HF model ID")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-5, help="Base learning rate (AdamW lr; Spectral-GD uses lr/5 on kto-mix-14k)")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--output_dir", type=str, default="./logs/finetune")
    p.add_argument("--run_name", type=str, default="")
    return p.parse_args()


def build_finetune_optimizers(model: torch.nn.Module, args):
    # For kto-mix-14k, Spectral-GD uses lr/5 as shown in Figure (b)
    eff_lr = (args.lr / 5.0) if (args.task == "kto-mix-14k" and args.optimizer == "spectral_gd") else args.lr

    if args.optimizer == "adamw":
        return [torch.optim.AdamW(model.parameters(), lr=eff_lr, weight_decay=args.weight_decay)]

    non_bias_params = []
    bias_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith(".bias") or "bias" in name:
            bias_params.append(p)
        else:
            non_bias_params.append(p)

    opts = [SpectralGD(non_bias_params, lr=eff_lr, weight_decay=args.weight_decay, rms_scale=False)]
    if bias_params:
        opts.append(torch.optim.AdamW(bias_params, lr=args.lr, weight_decay=0.0))
    return opts


@torch.no_grad()
def evaluate(model, test_loader, device, ctx):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    total_batches = 0

    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with ctx:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        valid_mask = shift_labels != -100
        preds = shift_logits.argmax(dim=-1)

        total_correct += (preds[valid_mask] == shift_labels[valid_mask]).sum().item()
        total_tokens += valid_mask.sum().item()
        total_loss += loss.item()
        total_batches += 1

    model.train()
    mean_loss = total_loss / max(1, total_batches)
    accuracy_pct = 100.0 * total_correct / max(1, total_tokens)
    return mean_loss, accuracy_pct


def main():
    args = parse_args()
    if not args.data_dir:
        args.data_dir = os.path.join("./data", args.task)
    if not args.model_name_or_path:
        args.model_name_or_path = DEFAULT_MODELS[args.task]
    if not args.run_name:
        args.run_name = f"{args.task}_{args.optimizer}_seed{args.seed}"

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_set = LocalJsonlDataset(os.path.join(args.data_dir, "train.jsonl"), tokenizer, args.max_length)
    test_set = LocalJsonlDataset(os.path.join(args.data_dir, "test.jsonl"), tokenizer, args.max_length)

    # Full-batch gradient accumulation over the fixed training dataset
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True).to(device)
    optimizers = build_finetune_optimizers(model, args)

    run_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    metrics_path = os.path.join(run_dir, "metrics.jsonl")

    num_micro_batches = len(train_loader)
    for epoch in range(1, args.epochs + 1):
        model.train()
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with ctx:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / max(1, num_micro_batches)
            loss.backward()

        for opt in optimizers:
            opt.step()

        test_loss, test_acc = evaluate(model, test_loader, device, ctx)
        record = {
            "epoch": epoch,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "task": args.task,
            "optimizer": args.optimizer,
            "seed": args.seed,
        }
        with open(metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"[{args.run_name}] Epoch {epoch}/{args.epochs} | Test loss: {test_loss:.4f} | Test acc: {test_acc:.2f}%")


if __name__ == "__main__":
    main()
