# Local Data Directory

Due to dataset size and redistribution constraints, the pretraining and fine-tuning datasets are omitted from this code archive and loaded from local paths under `./data/`.

Expected local directory structure:

```text
data/
├── fineweb/
│   ├── train.bin      # 70% training split of fixed FineWeb subset (character-level uint16 token IDs, V=65)
│   └── val.bin        # 30% validation split of fixed FineWeb subset (character-level uint16 token IDs, V=65)
├── gsm8k/
│   ├── train.jsonl    # GSM8K training split
│   └── test.jsonl     # GSM8K test split
└── kto-mix-14k/
    ├── train.jsonl    # kto-mix-14k training split
    └── test.jsonl     # kto-mix-14k test split
```
