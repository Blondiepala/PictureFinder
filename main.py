import argparse
import logging
import sys
import time
from pathlib import Path

from config import Config, select_device
from training import extract_training_embeddings
from detection import run_detection


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Find photos containing a specific person using face recognition.",
    )
    parser.add_argument(
        "--training", type=Path, default=Path("./data/training"),
        help="Folder with training images of the target person (default: ./data/training)",
    )
    parser.add_argument(
        "--detection", type=Path, default=Path("./data/detection"),
        help="Folder with images to search through (default: ./data/detection)",
    )
    parser.add_argument(
        "--results", type=Path, default=Path("./data/results"),
        help="Folder to copy matching images to (default: ./data/results)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.75,
        help="Cosine similarity threshold for matching (default: 0.75)",
    )
    parser.add_argument(
        "--max-size", type=int, default=1920,
        help="Max pixel size for longest image side during processing (default: 1920)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for embedding inference (default: 32)",
    )
    parser.add_argument(
        "--workers", type=int, default=0,
        help="Number of parallel face detection workers (default: auto = CPU count - 2)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose/debug logging",
    )

    args = parser.parse_args()
    return Config(
        training_dir=args.training,
        detection_dir=args.detection,
        results_dir=args.results,
        threshold=args.threshold,
        max_size=args.max_size,
        batch_size=args.batch_size,
        num_workers=args.workers,
        verbose=args.verbose,
    )


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )
    # Suppress noisy EXIF debug output from Pillow's JPEG parser
    logging.getLogger("PIL.TiffImagePlugin").setLevel(logging.INFO)


def main() -> None:
    config = parse_args()
    setup_logging(config.verbose)

    device = select_device()
    logging.info("Using device: %s", device)

    # Validate directories
    if not config.training_dir.is_dir():
        logging.error("Training directory does not exist: %s", config.training_dir)
        sys.exit(1)
    if not config.detection_dir.is_dir():
        logging.error("Detection directory does not exist: %s", config.detection_dir)
        sys.exit(1)

    # Phase 1: Training
    logging.info("Phase 1: Extracting training embeddings...")
    t0 = time.perf_counter()
    training_embeddings = extract_training_embeddings(config)
    t1 = time.perf_counter()
    logging.info("Training complete: %d embeddings in %.1fs", training_embeddings.shape[0], t1 - t0)

    # Phase 2: Detection
    logging.info("Phase 2: Scanning detection images...")
    t2 = time.perf_counter()
    match_count = run_detection(config, training_embeddings)
    t3 = time.perf_counter()
    logging.info("Detection complete: %d matches found in %.1fs", match_count, t3 - t2)

    logging.info("Results saved to %s", config.results_dir)
    logging.info("Total time: %.1fs", t3 - t0)


if __name__ == "__main__":
    main()
