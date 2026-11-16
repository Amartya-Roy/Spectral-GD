"""
Plotting script reproducing the paper figures:
- Fine-tuning plots (Figures gsm8k & ktomix):
    Top panel: Test loss vs. Epoch (1..5)
    Bottom panel: Test accuracy (%) vs. Epoch (1..5)
    Comparing AdamW (solid blue, circle markers) vs. Spectral-GD / Spectral-GD (lr/5) (dashed orange, square markers)
    with +/- 1 sigma shaded confidence bands across 3 seeds.
- Pretraining plots (Figures pythia & nanogpt):
    Left panel: Training Loss vs. Iteration k (0..5000)
    Right panel: Validation Loss vs. Iteration k (0..5000)
    Comparing Muon vs. Spectral-GD with +/- 1 sigma shaded confidence bands.
"""
import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def load_jsonl(path: str):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def plot_finetune(adamw_dirs, spectral_dirs, out_path: str, spectral_label: str = "Spectral-GD"):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.labelsize": 13,
        "legend.fontsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })

    def stack_finetune(run_dirs):
        losses, accs, epochs_ref = [], [], None
        for d in run_dirs:
            rows = sorted(load_jsonl(os.path.join(d, "metrics.jsonl")), key=lambda r: r["epoch"])
            ep = [r["epoch"] for r in rows]
            if epochs_ref is None:
                epochs_ref = ep
            losses.append([r["test_loss"] for r in rows])
            accs.append([r["test_acc"] for r in rows])
        return np.array(epochs_ref), np.array(losses), np.array(accs)

    epochs, adam_loss, adam_acc = stack_finetune(adamw_dirs)
    _, spec_loss, spec_acc = stack_finetune(spectral_dirs)

    fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(5.5, 5.2), sharex=True)

    styles = [
        ("AdamW", adam_loss, adam_acc, "#0072B2", "-", "o"),
        (spectral_label, spec_loss, spec_acc, "#D55E00", "--", "s"),
    ]

    for label, loss_arr, acc_arr, color, ls, marker in styles:
        loss_mean, loss_std = loss_arr.mean(axis=0), loss_arr.std(axis=0)
        acc_mean, acc_std = acc_arr.mean(axis=0), acc_arr.std(axis=0)

        ax_loss.plot(
            epochs, loss_mean, label=label, color=color, linestyle=ls, linewidth=2.0,
            marker=marker, markersize=5, markeredgecolor="white", markeredgewidth=0.8
        )
        ax_loss.fill_between(epochs, loss_mean - loss_std, loss_mean + loss_std, color=color, alpha=0.18)

        ax_acc.plot(
            epochs, acc_mean, label=label, color=color, linestyle=ls, linewidth=2.0,
            marker=marker, markersize=5, markeredgecolor="white", markeredgewidth=0.8
        )
        ax_acc.fill_between(epochs, acc_mean - acc_std, acc_mean + acc_std, color=color, alpha=0.18)

    ax_loss.set_ylabel("Test loss")
    ax_acc.set_ylabel("Test accuracy (%)")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_xticks(epochs)

    for ax in (ax_loss, ax_acc):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="-", alpha=0.25)

    handles, labels = ax_loss.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved fine-tuning plot to {out_path}")


def plot_pretrain(muon_dirs, spectral_dirs, out_path: str):
    colors = sns.color_palette("viridis_r", 2)
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    def stack_pretrain(run_dirs, split: str):
        all_losses = []
        steps_ref = None
        for d in run_dirs:
            rows = sorted(
                [r for r in load_jsonl(os.path.join(d, "metrics.jsonl")) if r["split"] == split],
                key=lambda r: r["step"],
            )
            steps = [r["step"] for r in rows]
            if steps_ref is None:
                steps_ref = steps
            all_losses.append([r["loss"] for r in rows])
        arr = np.array(all_losses)
        return np.array(steps_ref), arr.mean(axis=0), arr.std(axis=0)

    fig, (ax_train, ax_val) = plt.subplots(1, 2, figsize=(12, 5))
    for ax, split, title in ((ax_train, "train", "Training Loss"), (ax_val, "val", "Validation Loss")):
        for label, dirs, color in (("Muon", muon_dirs, colors[0]), ("Spectral-GD", spectral_dirs, colors[1])):
            steps, mean, std = stack_pretrain(dirs, split)
            ax.plot(steps, mean, label=label, color=color, linewidth=2.0)
            ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.2)
        ax.set_xlabel(r"Iteration $k$")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(frameon=True)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved pretraining plot to {out_path}")


def main():
    p = argparse.ArgumentParser(description="Plot Fine-tuning and Pretraining Comparisons")
    p.add_argument("--mode", choices=["finetune", "pretrain"], required=True)
    p.add_argument("--baseline", nargs="+", required=True, help="Run dirs for AdamW (finetune) or Muon (pretrain)")
    p.add_argument("--spectral_gd", nargs="+", required=True, help="Run dirs for Spectral-GD")
    p.add_argument("--spectral_label", type=str, default="Spectral-GD", help="Legend label (e.g. 'Spectral-GD (lr/5)')")
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    if args.mode == "finetune":
        plot_finetune(args.baseline, args.spectral_gd, args.out, spectral_label=args.spectral_label)
    else:
        plot_pretrain(args.baseline, args.spectral_gd, args.out)


if __name__ == "__main__":
    main()
