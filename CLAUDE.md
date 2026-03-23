# Parameter Golf Competition

## What This Is
OpenAI Model Craft Challenge: Parameter Golf. Train the best LM under 16MB artifact + 10min on 8xH100s.
Deadline: April 30, 2026. GitHub user: channyzf6.

## Constraints
- Artifact = code bytes (`train_gpt.py`) + compressed model bytes <= 16,000,000 bytes (decimal)
- Training: max 10 minutes on 8xH100 SXM
- Evaluation: max 10 additional minutes on 8xH100
- No network/external data during evaluation
- Must be fully self-contained and reproducible

## Metric
`val_bpb` (bits-per-byte) on FineWeb validation set (first 50k docs). Lower = better.
Formula: `val_bpb = (val_loss / ln(2)) * (tokens / bytes)` — tokenizer-agnostic.

## Current SOTA (as of 2026-03-23)
1.1428 bpb by thwu1 — 10L Int5-MLP + BigramHash(10240) + SWA(0.4) + WD=0.04

## To Beat SOTA
Must improve by >= 0.005 nats, with p < 0.01 significance (typically 3-run average).

## Submission Format
PR to `records/track_10min_16mb/` with:
- README.md (technical explanation)
- submission.json (name, github_id, val_bpb, metadata)
- Training logs (3 runs for significance)
- Working train_gpt.py + dependencies

## Key Architecture (Baseline)
- 9 layers, 512 dim, 8 heads, 4 KV heads (GQA), 2x MLP, 1024 vocab
- Tied embeddings, U-Net skip connections, RoPE, logit softcap=30
- Muon optimizer (matrices), Adam (embeddings/scalars)
- ReLU^2 MLP, RMSNorm, CastedLinear (fp32 weights, bf16 compute)
- Baseline score: 1.2244 bpb

## Winning Techniques (from leaderboard analysis)
1. **MLP 3x expansion** (1536 hidden) — universal across top entries
2. **Quantization**: Int5/Int6 mixed > STE QAT Int6 > Post-training Int8
3. **BigramHash**: 4096-10240 buckets, dim=128, hash-based bigram embeddings
4. **SmearGate**: Per-dimension sigmoid gate blending current/previous token embeddings
5. **Sliding window eval**: stride=64, significantly improves val_bpb
6. **Longer sequences**: 2048 > 1024 for both train and eval
7. **SWA** (Stochastic Weight Averaging): last 40-50% of training
8. **Weight decay**: 0.04 on Muon, improves quantization friendliness
9. **Orthogonal init**: gain=1.0, output projections scaled 1/sqrt(2*layers)
10. **zstd-22** compression (better than zlib-9 on quantized data)

## Workflow
1. Edit code locally in this repo
2. Push to GitHub fork
3. SSH into RunPod pod (8xH100)
4. Pull, download data, train, analyze logs
5. Iterate

## Commands
```bash
# Download data
python3 data/cached_challenge_fineweb.py --variant sp1024

# Train (8xH100)
RUN_ID=my_run DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py

# Train (1xH100, for testing)
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## File Structure
- `train_gpt.py` — main training script (this is what you modify)
- `train_gpt_mlx.py` — Apple Silicon variant for local iteration
- `data/` — dataset download scripts, tokenizers, cached shards
- `records/track_10min_16mb/` — leaderboard submissions
- `records/track_non_record_16mb/` — non-competitive submissions
