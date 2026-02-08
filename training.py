import logging
from pathlib import Path

import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

from config import Config, select_device
from utils import list_images, load_image

logger = logging.getLogger(__name__)

CACHE_FILE = "embeddings_cache.pt"


def create_models(device: torch.device) -> tuple[MTCNN, MTCNN, InceptionResnetV1]:
    """Create and return training MTCNN, detection MTCNN, and InceptionResnetV1 embedder.

    MTCNN runs on CPU (its multi-stage pipeline is faster on CPU than MPS).
    The embedder runs on the target device for batch GPU acceleration.
    """
    mtcnn_training = MTCNN(
        select_largest=True,
        device=torch.device("cpu"),
        post_process=True,
    )
    mtcnn_detection = MTCNN(
        keep_all=True,
        device=torch.device("cpu"),
        post_process=True,
    )
    embedder = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    return mtcnn_training, mtcnn_detection, embedder


def _cache_path(training_dir: Path) -> Path:
    return training_dir / CACHE_FILE


def _load_cache(training_dir: Path, image_names: set[str]) -> torch.Tensor | None:
    """Load cached embeddings if they match the current training file set."""
    path = _cache_path(training_dir)
    if not path.exists():
        return None

    try:
        cache = torch.load(path, weights_only=False)
    except Exception as e:
        logger.warning("Cache corrupted, regenerating: %s", e)
        return None

    if cache.get("filenames") != image_names:
        logger.info("Training files changed, regenerating embeddings")
        return None

    logger.info("Loaded cached embeddings (%d faces)", cache["embeddings"].shape[0])
    return cache["embeddings"]


def _save_cache(training_dir: Path, embeddings: torch.Tensor, image_names: set[str]) -> None:
    torch.save({"filenames": image_names, "embeddings": embeddings}, _cache_path(training_dir))
    logger.info("Saved embeddings cache to %s", _cache_path(training_dir))


def extract_training_embeddings(config: Config) -> torch.Tensor:
    """Extract face embeddings from training images. Returns tensor of shape (N, 512)."""
    images = list_images(config.training_dir)
    if not images:
        raise ValueError(f"No images found in {config.training_dir}")

    image_names = {p.name for p in images}

    # Try cache first
    cached = _load_cache(config.training_dir, image_names)
    if cached is not None:
        return cached

    device = select_device()
    mtcnn_training, _, embedder = create_models(device)

    embeddings = []
    for img_path in images:
        img = load_image(img_path, config.max_size)
        if img is None:
            continue

        face = mtcnn_training(img)  # returns tensor (3, 160, 160) or None
        if face is None:
            logger.warning("No face detected in training image: %s", img_path.name)
            continue

        face = face.unsqueeze(0).to(device)  # (1, 3, 160, 160)
        with torch.no_grad():
            emb = embedder(face)  # (1, 512)
        embeddings.append(emb.cpu())
        logger.debug("Extracted embedding from %s", img_path.name)

    if not embeddings:
        raise RuntimeError(
            "No faces detected in any training image. "
            "Ensure training images contain clear, visible faces."
        )

    result = torch.cat(embeddings, dim=0)  # (N, 512)
    logger.info("Extracted %d embeddings from %d training images", result.shape[0], len(images))

    _save_cache(config.training_dir, result, image_names)
    return result
