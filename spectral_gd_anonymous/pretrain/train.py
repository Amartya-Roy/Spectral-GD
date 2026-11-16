"""
Full-batch pretraining from random initialization on a fixed FineWeb subset:
- Models:
    * pythia: Pythia-160M (12 layers, model dimension 768)
    * nanogpt: modded-nanogpt (11 layers, model dimension 768)
- Optimizers:
    * spectral_gd: Full-batch Spectral-GD; updates all parameters except biases.
    * muon: Full-batch Muon.
- Runs for 5,000 iterations over a fixed FineWeb subset stored locally under ./data/fineweb.
"""
import os
import sys
import json
import math
import time
import argparse
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, all_reduce, ReduceOp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from optimizers import SpectralGD, Muon
from pretrain.models import Pythia160MConfig, Pythia160M, ModdedNanoGPTConfig, ModdedNanoGPT


def parse_args():
    p = argparse.ArgumentParser(description="Full-batch Pretraining on FineWeb (Pythia-160M & Modded-NanoGPT)")
    p.add_argument("--model", type=str, choices=["pythia", "nanogpt"], required=True)
    p.add_argument("--optimizer", type=str, choices=["muon", "spectral_gd"], required=True)
    p.add_argument("--data_dir", type=str, default="./data/fineweb")
    p.add_argument("--subset_tokens", type=int, default=524288, help="Fixed FineWeb training subset size in tokens")
    p.add_argument("--val_tokens", type=int, default=524288, help="Fixed FineWeb validation subset size in tokens")
    p.add_argument("--block_size", type=int, default=1024)
    p.add_argument("--micro_batch_size", type=int, default=16, help="Micro-batch size per forward/backward pass")
    p.add_argument("--num_iterations", type=int, default=5000)
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--lr", type=float, default=0.005, help="Learning rate for Spectral-GD / Muon")
    p.add_argument("--bias_lr", type=float, default=3e-4, help="Learning rate for bias parameters (if any)")
    p.add_argument("--muon_momentum", type=float, default=0.95)
    p.add_argument("--warmup_iters", type=int, default=250)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--output_dir", type=str, default="./logs/pretrain")
    p.add_argument("--run_name", type=str, default="")
    return p.parse_args()


def load_fixed_subset(bin_path: str, num_tokens: int, block_size: int):
    """Loads a fixed deterministic set of contiguous sequences covering the entire subset."""
    if not os.path.exists(bin_path):
        raise FileNotFoundError(
            f"Local FineWeb binary file not found at {bin_path}. "
            f"Please place train.bin and val.bin inside ./data/fineweb/."
        )
    data = np.memmap(bin_path, dtype=np.uint16, mode="r")
    num_sequences = num_tokens // block_size
    xs, ys = [], []
    for i in range(num_sequences):
        start = i * block_size
        chunk = data[start : start + block_size + 1].astype(np.int64)
        xs.append(torch.from_numpy(chunk[:-1].copy()))
        ys.append(torch.from_numpy(chunk[1:].copy()))
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0)


def build_model(model_name: str, block_size: int):
    if model_name == "pythia":
        cfg = Pythia160MConfig(n_layer=12, n_head=12, n_embd=768, block_size=block_size, bias=True)
        return Pythia160M(cfg)
    elif model_name == "nanogpt":
        cfg = ModdedNanoGPTConfig(n_layer=11, n_head=6, n_embd=768, block_size=block_size)
        return ModdedNanoGPT(cfg)
    raise ValueError(f"Unknown model: {model_name}")


def build_optimizers(model: torch.nn.Module, args):
    """
    Spectral-GD (and Muon) updates all parameters except the biases.
    Any bias parameters (present in Pythia-160M) are separated into a dedicated bias group.
    """
    non_bias_params = []
    bias_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith(".bias") or "bias" in name:
            bias_params.append(p)
        else:
            non_bias_params.append(p)

    optimizers = []
    if args.optimizer == "spectral_gd":
        optimizers.append(SpectralGD(non_bias_params, lr=args.lr))
    elif args.optimizer == "muon":
        optimizers.append(Muon(non_bias_params, lr=args.lr, momentum=args.muon_momentum))

    if len(bias_params) > 0:
        optimizers.append(torch.optim.AdamW(bias_params, lr=args.bias_lr, weight_decay=0.0))

    return optimizers


def main():
    args = parse_args()
    if not args.run_name:
        args.run_name = f"{args.model}_{args.optimizer}_seed{args.seed}"

    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        master_process = rank == 0
    else:
        rank, local_rank, world_size = 0, 0, 1
        master_process = True
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device_type = "cuda" if "cuda" in device else "cpu"
    torch.manual_seed(args.seed)
    if device_type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)

    train_X, train_Y = load_fixed_subset(
        os.path.join(args.data_dir, "train.bin"), args.subset_tokens, args.block_size
    )
    val_X, val_Y = load_fixed_subset(
        os.path.join(args.data_dir, "val.bin"), args.val_tokens, args.block_size
    )

    # Partition fixed dataset across DDP ranks for exact full-batch gradient accumulation
    per_rank_seqs = train_X.size(0) // world_size
    rank_train_X = train_X[rank * per_rank_seqs : (rank + 1) * per_rank_seqs]
    rank_train_Y = train_Y[rank * per_rank_seqs : (rank + 1) * per_rank_seqs]
    num_micro_steps = math.ceil(rank_train_X.size(0) / args.micro_batch_size)

    model = build_model(args.model, args.block_size).to(device)
    if ddp:
        model = DDP(model, device_ids=[local_rank])
    raw_model = model.module if ddp else model

    optimizers = build_optimizers(raw_model, args)

    def lr_schedule(step: int) -> float:
        if step < args.warmup_iters:
            return (step + 1) / max(1, args.warmup_iters)
        progress = (step - args.warmup_iters) / max(1, args.num_iterations - args.warmup_iters)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine

    schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, lr_schedule) for opt in optimizers]

    run_dir = os.path.join(args.output_dir, args.run_name)
    if master_process:
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
    metrics_path = os.path.join(run_dir, "metrics.jsonl")

    def log_metric(step: int, split: str, loss_val: float):
        if master_process:
            with open(metrics_path, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "step": step,
                            "split": split,
                            "loss": loss_val,
                            "model": args.model,
                            "optimizer": args.optimizer,
                            "seed": args.seed,
                        }
                    )
                    + "\n"
                )

    @torch.no_grad()
    def evaluate_full_split(X_split: torch.Tensor, Y_split: torch.Tensor) -> float:
        model.eval()
        n_seqs = X_split.size(0)
        n_batches = math.ceil(n_seqs / args.micro_batch_size)
        total_loss = 0.0
        for b in range(n_batches):
            xb = X_split[b * args.micro_batch_size : (b + 1) * args.micro_batch_size].to(device)
            yb = Y_split[b * args.micro_batch_size : (b + 1) * args.micro_batch_size].to(device)
            with ctx:
                _, loss = model(xb, yb)
            total_loss += loss.item() * (xb.size(0) / n_seqs)
        model.train()
        return total_loss

    for step in range(args.num_iterations + 1):
        if step % args.eval_interval == 0 or step == args.num_iterations:
            val_loss = evaluate_full_split(val_X, val_Y)
            log_metric(step, "val", val_loss)
            if master_process:
                print(f"[{args.run_name}] iter {step}/{args.num_iterations} | val_loss: {val_loss:.4f}")

        if step == args.num_iterations:
            break

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        # Full-batch gradient computation over the entire fixed training subset via microbatch accumulation
        step_loss = 0.0
        for micro_step in range(num_micro_steps):
            if ddp:
                model.require_backward_grad_sync = micro_step == (num_micro_steps - 1)
            s_idx = micro_step * args.micro_batch_size
            e_idx = min(rank_train_X.size(0), (micro_step + 1) * args.micro_batch_size)
            weight = (e_idx - s_idx) / rank_train_X.size(0)

            xb = rank_train_X[s_idx:e_idx].to(device)
            yb = rank_train_Y[s_idx:e_idx].to(device)
            with ctx:
                _, loss = model(xb, yb)
                scaled_loss = loss * weight
            scaled_loss.backward()
            step_loss += scaled_loss.item()

        if ddp:
            t_loss = torch.tensor(step_loss, device=device)
            all_reduce(t_loss, op=ReduceOp.AVG)
            step_loss = t_loss.item()

        for opt, sched in zip(optimizers, schedulers):
            opt.step()
            sched.step()

        log_metric(step + 1, "train", step_loss)
        if master_process and (step + 1) % 50 == 0:
            print(f"[{args.run_name}] iter {step + 1}/{args.num_iterations} | train_loss: {step_loss:.4f}")

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
