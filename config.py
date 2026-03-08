from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class Config:
    training_dir: Path = Path("./data/training")
    detection_dir: Path = Path("./data/detection")
    results_dir: Path = Path("./data/results")
    threshold: float = 0.75
    max_size: int = 1920
    batch_size: int = 32
    num_workers: int = 0  # 0 = auto (CPU count - 2)
    verbose: bool = False
    video_path: Path | None = None
    video_start: float | None = None  # seconds
    video_end: float | None = None  # seconds


def select_device() -> torch.device:
    """Select best available device: MPS → CUDA → CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
