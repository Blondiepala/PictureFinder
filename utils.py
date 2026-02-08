import logging
import shutil
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def list_images(folder: Path) -> list[Path]:
    """Return sorted list of image files in folder."""
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_image(path: Path, max_size: int) -> Image.Image | None:
    """Load an image and resize so longest side <= max_size. Returns None on failure."""
    try:
        img = Image.open(path).convert("RGB")
    except Exception as e:
        logger.warning("Could not load %s: %s", path.name, e)
        return None

    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    return img


def copy_file(src: Path, dest_dir: Path) -> Path:
    """Copy file to dest_dir, appending numeric suffix on collision."""
    dest = dest_dir / src.name
    if not dest.exists():
        shutil.copy2(src, dest)
        return dest

    stem = src.stem
    suffix = src.suffix
    counter = 1
    while dest.exists():
        dest = dest_dir / f"{stem}_{counter}{suffix}"
        counter += 1
    shutil.copy2(src, dest)
    return dest
