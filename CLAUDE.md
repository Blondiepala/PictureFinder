# CLAUDE.md - Project Guide for Claude Code

## Project Overview

PictureFinder is a Python face recognition tool that finds photos and video segments containing a specific person. It uses facenet-pytorch (MTCNN + InceptionResnetV1/VGGFace2) for face detection and embedding similarity matching. Supports both image folder scanning and video timeline detection.

## Environment

- **Conda env**: `picturefinder` (Python 3.11)
- **Python path**: `/Users/viktor/miniforge3/envs/picturefinder/bin/python`
- **Platform**: macOS Apple Silicon (MPS available)
- **Run command**: `/Users/viktor/miniforge3/envs/picturefinder/bin/python main.py`

## Architecture

```
main.py        → CLI (argparse), logging, two-phase orchestration (image/video mode)
config.py      → Config dataclass, select_device() (MPS → CUDA → CPU)
training.py    → create_models(), extract_training_embeddings() with cache
detection.py   → run_detection() with multiprocessing face detection
video.py       → run_video_detection() streaming frame-by-frame video scanning
utils.py       → list_images(), load_image(), copy_file()
```

### Pipeline

1. **Training**: Load images → MTCNN (select_largest) → InceptionResnetV1 → 512-dim embeddings → cache to `embeddings_cache.pt`
2. **Image Detection**: Parallel MTCNN workers (CPU) → batch embeddings on MPS → cosine similarity → copy matches
3. **Video Detection**: Stream frames via OpenCV → sample at 4fps → MTCNN (CPU, min_face_size=15) → batch embeddings on MPS → cosine similarity → merge timestamps into segments → write `HH:MM:SS - HH:MM:SS` output file

### Key Design Decisions

- **MTCNN runs on CPU** — MPS `adaptive_avg_pool2d` doesn't support non-divisible sizes. CPU is also faster for MTCNN's multi-stage pipeline than MPS due to transfer overhead.
- **Embedder (InceptionResnetV1) runs on MPS/GPU** — batch processing benefits from GPU.
- **Face detection is parallelized** via `ProcessPoolExecutor` with `num_workers` (default: CPU count - 2). Each worker creates its own MTCNN instance with `torch.set_num_threads(1)`.
- **Default threshold is 0.75** — tuned for VGGFace2 embeddings. True positives typically score 0.85+, false positives below 0.70.
- **Embedding cache** stored in training folder, validated by comparing filename sets.
- **Video frames resized to 1080px** — balances memory usage with detecting smaller/distant faces.
- **Video segment merging** — consecutive detections within 0.75s merged; segments < 1s discarded to filter false positives.

## Data Layout

```
data/
├── training/              # ~10-20 images of target person
│   └── embeddings_cache.pt  # auto-generated cache
├── detection/             # images to search (can be 1000+)
├── results/               # matched originals copied here
└── video/                 # video files to scan
```

## Testing Changes

After any code change, verify with:
```bash
# Image mode test
/Users/viktor/miniforge3/envs/picturefinder/bin/python main.py -v

# Check results count
ls data/results/ | wc -l

# Video mode test (with time range)
/Users/viktor/miniforge3/envs/picturefinder/bin/python main.py --video data/video/example.mp4 --video-start 1:00 --video-end 2:00 -v

# Check video output
cat data/video/example_segments.txt
```

Delete `data/training/embeddings_cache.pt` to force re-training after modifying training logic.

## Common Pitfalls

- Don't put MTCNN on MPS — it will be 4x slower or crash
- Pillow's `PIL.TiffImagePlugin` logger is noisy at DEBUG level — already suppressed in `setup_logging()`
- `ProcessPoolExecutor` workers need self-contained imports (no shared state)
- Image extensions supported: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tiff`, `.tif`
- Video detection uses `opencv-python<4.11` for numpy<2 compatibility with facenet-pytorch
- MPS cache must be cleared periodically during video processing to prevent memory buildup
