# CLAUDE.md - Project Guide for Claude Code

## Project Overview

PictureFinder is a Python face recognition tool that finds photos containing a specific person. It uses facenet-pytorch (MTCNN + InceptionResnetV1/VGGFace2) for face detection and embedding similarity matching.

## Environment

- **Conda env**: `picturefinder` (Python 3.11)
- **Python path**: `/Users/viktor/miniforge3/envs/picturefinder/bin/python`
- **Platform**: macOS Apple Silicon (MPS available)
- **Run command**: `/Users/viktor/miniforge3/envs/picturefinder/bin/python main.py`

## Architecture

```
main.py        → CLI (argparse), logging, two-phase orchestration
config.py      → Config dataclass, select_device() (MPS → CUDA → CPU)
training.py    → create_models(), extract_training_embeddings() with cache
detection.py   → run_detection() with multiprocessing face detection
utils.py       → list_images(), load_image(), copy_file()
```

### Pipeline

1. **Training**: Load images → MTCNN (select_largest) → InceptionResnetV1 → 512-dim embeddings → cache to `embeddings_cache.pt`
2. **Detection**: Parallel MTCNN workers (CPU) → batch embeddings on MPS → cosine similarity → copy matches

### Key Design Decisions

- **MTCNN runs on CPU** — MPS `adaptive_avg_pool2d` doesn't support non-divisible sizes. CPU is also faster for MTCNN's multi-stage pipeline than MPS due to transfer overhead.
- **Embedder (InceptionResnetV1) runs on MPS/GPU** — batch processing benefits from GPU.
- **Face detection is parallelized** via `ProcessPoolExecutor` with `num_workers` (default: CPU count - 2). Each worker creates its own MTCNN instance with `torch.set_num_threads(1)`.
- **Default threshold is 0.75** — tuned for VGGFace2 embeddings. True positives typically score 0.85+, false positives below 0.70.
- **Embedding cache** stored in training folder, validated by comparing filename sets.

## Data Layout

```
data/
├── training/              # ~10-20 images of target person
│   └── embeddings_cache.pt  # auto-generated cache
├── detection/             # images to search (can be 1000+)
└── results/               # matched originals copied here
```

## Testing Changes

After any code change, verify with:
```bash
# Quick test (small set if available, otherwise full)
/Users/viktor/miniforge3/envs/picturefinder/bin/python main.py -v

# Check results count
ls data/results/ | wc -l
```

Delete `data/training/embeddings_cache.pt` to force re-training after modifying training logic.

## Common Pitfalls

- Don't put MTCNN on MPS — it will be 4x slower or crash
- Pillow's `PIL.TiffImagePlugin` logger is noisy at DEBUG level — already suppressed in `setup_logging()`
- `ProcessPoolExecutor` workers need self-contained imports (no shared state)
- Image extensions supported: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tiff`, `.tif`
