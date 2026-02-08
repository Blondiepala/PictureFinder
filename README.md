# PictureFinder

Automatically find photos containing a specific person. Provide ~10 training images of the target person and a folder of images to search through. Matching images are copied (originals, full-resolution) to a results folder.

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

### Quick start

```bash
mkdir -p data/training data/detection data/results

# Add ~10 photos of the target person to data/training/
# Add photos to search through to data/detection/

python main.py -v
```

### Threshold tuning

- **0.75** (default) - strict, minimizes false positives
- **0.65** - moderate, catches more matches but may include lookalikes
- **0.55** - permissive, use when the target appears at difficult angles or lighting

## How it works

**Phase 1 - Training:** Load training images, detect the largest face in each (MTCNN), extract 512-dimensional embeddings (InceptionResnetV1). Embeddings are cached in `embeddings_cache.pt` and reused on subsequent runs.

**Phase 2 - Detection:** Detect all faces in each image using parallel workers, batch face crops through the embedder, compute cosine similarity against training embeddings. If any face exceeds the threshold, the original full-resolution image is copied to results.

## Project structure

```
PictureFinder/
├── main.py           # CLI entry point and orchestration
├── config.py         # Config dataclass and device selection
├── training.py       # Embedding extraction and cache management
├── detection.py      # Parallel face detection and similarity matching
├── utils.py          # Image I/O, resizing, file copy helpers
└── requirements.txt  # Dependencies
```
