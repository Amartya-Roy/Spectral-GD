# Spectral-GD vs. AdamW and Muon: Fine-Tuning and Pretraining Benchmarks

Anonymous code repository accompanying the paper's experimental evaluation of **Spectral-GD** (deterministic polar-factor gradient descent via exact SVD) against **AdamW** (on language-model fine-tuning) and **Muon** (on full-batch language-model pretraining).

## Repository Structure

```text
.
├── data/
│   ├── README.md          # Expected format for local dataset files
│   ├── fineweb/           # Local FineWeb pretraining subset (train.bin, val.bin)
│   ├── gsm8k/             # Local GSM8K dataset (train.jsonl, test.jsonl)
│   └── kto-mix-14k/       # Local kto-mix-14k dataset (train.jsonl, test.jsonl)
├── finetune/
│   └── train.py           # Fine-tuning Qwen-3 1.7B (GSM8K) and Qwen-1.5 0.5B (kto-mix-14k)
├── pretrain/
│   ├── models.py          # Pythia-160M (12L, d=768) and Modded-NanoGPT (11L, d=768)
│   └── train.py           # Full-batch pretraining from random initialization on FineWeb
├── optimizers.py          # Spectral-GD (exact SVD polar factor) and Muon (Newton-Schulz)
├── plot_results.py        # Generates the fine-tuning and pretraining figures
├── requirements.txt       # Python package dependencies
└── run_experiments.sh     # End-to-end launcher across 3 seeds and figure generation
```

## Datasets (Local Data Path)

Datasets are omitted from this archive due to size constraints and are loaded directly from `./data/`:
- `./data/gsm8k/{train.jsonl, test.jsonl}`
- `./data/kto-mix-14k/{train.jsonl, test.jsonl}`
- `./data/fineweb/{train.bin, val.bin}`

## Experiments

### 1. Fine-Tuning (Figures `fig_gsm8k.png` and `fig_kto-mix-14k.png`)
Compares **Spectral-GD** with **AdamW** over 5 epochs and 3 seeds:
- **GSM8K** with **Qwen-3 1.7B** (`--task gsm8k`)
- **kto-mix-14k** with **Qwen-1.5 0.5B** (`--task kto-mix-14k`, where Spectral-GD uses `lr/5`)

```bash
python finetune/train.py --task gsm8k --optimizer spectral_gd --data_dir ./data/gsm8k --epochs 5 --seed 1
python finetune/train.py --task kto-mix-14k --optimizer spectral_gd --data_dir ./data/kto-mix-14k --epochs 5 --seed 1
```

### 2. Pretraining (Figures `fig_pythia.png` and `fig_nanogpt.png`)
Compares full-batch **Spectral-GD** against full-batch **Muon** over 5,000 iterations from random initialization on the fixed FineWeb subset (`./data/fineweb`), where Spectral-GD updates all parameters except the biases:
- **Pythia-160M** (12 layers, model dimension 768): `--model pythia`
- **Modded-NanoGPT** (11 layers, model dimension 768): `--model nanogpt`

```bash
python pretrain/train.py --model pythia --optimizer spectral_gd --data_dir ./data/fineweb --num_iterations 5000 --seed 1
python pretrain/train.py --model nanogpt --optimizer spectral_gd --data_dir ./data/fineweb --num_iterations 5000 --seed 1
```

### Reproducing All Runs and Plots

```bash
bash run_experiments.sh
```
