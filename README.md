# PictureFinder

Automatically find photos and video segments containing a specific person. Provide ~10 training images of the target person and a folder of images (or a video file) to search through. Matching images are copied (originals, full-resolution) to a results folder; matching video moments are written as timestamped segments.

Uses **facenet-pytorch** for face detection (MTCNN) and face embedding (InceptionResnetV1/VGGFace2). Works 100% locally, no uploads.

## Setup

```bash
conda create -n picturefinder python=3.11 -y
conda activate picturefinder
pip install -r requirements.txt
```

## Usage

```bash
python main.py [--training FOLDER] [--detection FOLDER] [--results FOLDER]
               [--threshold 0.75] [--max-size 1920] [--batch-size 32]
               [--workers N] [-v]
               [--video FILE] [--video-start TIME] [--video-end TIME]
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--training` | `./data/training` | Folder with training images of the target person |
| `--detection` | `./data/detection` | Folder with images to search through |
| `--results` | `./data/results` | Folder to copy matching images to |
| `--threshold` | `0.75` | Cosine similarity threshold (lower = more matches) |
| `--max-size` | `1920` | Max pixel size for longest side during processing |
| `--batch-size` | `32` | Batch size for embedding inference |
| `--workers` | auto | Parallel face detection workers (default: CPU count - 2) |
| `-v` | off | Verbose/debug logging |
| `--video` | — | Path to a video file to scan (activates video mode) |
| `--video-start` | beginning | Start time for video scan, e.g. `1:30` or `1:05:30` |
| `--video-end` | end | End time for video scan, e.g. `2:00` or `1:10:00` |

### Quick start — image mode

```bash
mkdir -p data/training data/detection data/results

# Add ~10 photos of the target person to data/training/
# Add photos to search through to data/detection/

python main.py -v
```

### Quick start — video mode

```bash
mkdir -p data/training data/video

# Add ~10 photos of the target person to data/training/
# Place your video in data/video/

# Scan entire video
python main.py --video data/video/example.mp4 -v

# Scan a specific time range
python main.py --video data/video/example.mp4 --video-start 1:00 --video-end 2:00 -v
```

Results are written to `{video_stem}_segments.txt` next to the video file, one segment per line:

```
00:01:03 - 00:01:47
00:03:12 - 00:03:55
```

### Threshold tuning

- **0.75** (default) - strict, minimizes false positives
- **0.65** - moderate, catches more matches but may include lookalikes
- **0.55** - permissive, use when the target appears at difficult angles or lighting

## How it works

**Phase 1 - Training:** Load training images, detect the largest face in each (MTCNN), extract 512-dimensional embeddings (InceptionResnetV1). Embeddings are cached in `embeddings_cache.pt` and reused on subsequent runs.

**Phase 2 - Image detection:** Detect all faces in each image using parallel workers, batch face crops through the embedder, compute cosine similarity against training embeddings. If any face exceeds the threshold, the original full-resolution image is copied to results.

**Phase 2 - Video detection:** Stream frames from the video at ~4 fps, resize to 1080px, run MTCNN face detection, batch embeddings on GPU, and compute cosine similarity. Matched timestamps are merged into continuous segments (gaps ≤ 0.5 s are bridged; segments < 1 s are discarded as false positives). Segments are written to a `{video_stem}_segments.txt` file in `HH:MM:SS - HH:MM:SS` format.

## Project structure

```
PictureFinder/
├── main.py           # CLI entry point and orchestration
├── config.py         # Config dataclass and device selection
├── training.py       # Embedding extraction and cache management
├── detection.py      # Parallel face detection and similarity matching
├── video.py          # Video scanning, segment merging, and output writing
├── utils.py          # Image I/O, resizing, file copy helpers
└── requirements.txt  # Dependencies
```
