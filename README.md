# My First AI v5 BPE training path

This package is a real PyTorch training path for the custom decoder-only Transformer. It does not call an external tokenizer or model API.

## 1. Prepare data

```bash
python training/prepare_data.py --input data/train.txt --out data/processed --vocab-size 1024
```

This produces `tokenizer.json`, `train.bin`, `val.bin`, and `manifest.json`. The tokenizer starts from UTF-8 bytes and learns frequent adjacent byte-pair merges into subword tokens.

## 2. Train on CUDA

```bash
python training/train.py --data-dir data/processed --steps 1000 --batch-size 32 --lr 0.0003 --seed 42 --save-every 500 --eval-every 100 --out checkpoints/my-first-ai-v5-bpe.pt
```

The trainer uses mini-batches, AdamW, gradient clipping, mixed precision on CUDA, deterministic seed initialization, sampled training windows, deterministic validation windows, validation loss, perplexity, periodic checkpoints, and a final checkpoint.

## 3. Resume training

`--steps` means additional optimizer steps. Resume verifies the checkpoint architecture and model dimensions before loading weights and optimizer state.

```bash
python training/train.py --data-dir data/processed --resume checkpoints/my-first-ai-v5-bpe.pt --steps 1000 --batch-size 32 --lr 0.0003 --seed 42 --out checkpoints/my-first-ai-v5-bpe.pt
```

Checkpoints include architecture, model configuration, exact parameter count, model state, AdamW state, CUDA scaler state when applicable, step, best validation loss, tokenizer vocabulary, and dataset manifest metadata.

## 4. Evaluate a checkpoint

```bash
python training/evaluate.py --checkpoint checkpoints/my-first-ai-v5-bpe.pt --data-dir data/processed
```

This reports validation loss, perplexity, checkpoint step, parameter count, and tokenizer vocabulary.

## Current v5 architecture

- BPE over UTF-8 bytes
- vocabulary from 256 byte IDs up to the learned target vocabulary
- dModel 64
- 4 decoder blocks
- context 256
- feed-forward width 256
- exact parameter count at vocab 1024: 345,088
- AdamW
- gradient clipping at 1.0
- float16 autocast on CUDA

## Compatibility

The existing hosted v4 byte-level model remains the working application baseline. v5 is a separate checkpoint because changing the tokenizer changes the embedding and output vocabulary dimensions.

GPU training is real when this package is run on a CUDA machine. The hosted application/API layer does not claim that its CPU runtime has trained the larger v5 model on a GPU.
