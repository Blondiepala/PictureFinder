import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from tqdm import tqdm

from config import Config, select_device
from utils import list_images, load_image, copy_file

logger = logging.getLogger(__name__)


def _detect_faces_in_chunk(args: tuple) -> list[tuple[str, torch.Tensor]]:
    """Worker: run MTCNN on a chunk of images. Returns list of (filename, face_crops)."""
    image_paths, max_size = args

    mtcnn = MTCNN(keep_all=True, device=torch.device("cpu"), post_process=True)
    # Limit per-worker threads to avoid oversubscription
    torch.set_num_threads(1)

    results = []
    for img_path in image_paths:
        img = load_image(img_path, max_size)
        if img is None:
            continue

        faces = mtcnn(img)
        if faces is None:
            continue

        if faces.dim() == 3:
            faces = faces.unsqueeze(0)

        results.append((str(img_path), faces.cpu()))

    return results


def _get_num_workers(config: Config) -> int:
    if config.num_workers > 0:
        return config.num_workers
    return max(1, os.cpu_count() - 2)


def run_detection(
    config: Config,
    training_embeddings: torch.Tensor,
) -> int:
    """Run detection on all images. Returns number of matches found."""
    images = list_images(config.detection_dir)
    if not images:
        logger.warning("No images found in %s", config.detection_dir)
        return 0

    config.results_dir.mkdir(parents=True, exist_ok=True)

    device = select_device()
    embedder = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    training_embeddings = training_embeddings.to(device)

    num_workers = _get_num_workers(config)
    logger.info("Face detection using %d workers", num_workers)

    # Split images into chunks for workers
    chunk_size = max(1, len(images) // num_workers)
    chunks = []
    for i in range(0, len(images), chunk_size):
        chunks.append((images[i : i + chunk_size], config.max_size))

    # Phase 1: Parallel face detection on CPU
    all_detections: list[tuple[str, torch.Tensor]] = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_detect_faces_in_chunk, chunk): i for i, chunk in enumerate(chunks)}
        with tqdm(total=len(images), desc="Detecting faces", unit="img") as pbar:
            for future in as_completed(futures):
                chunk_results = future.result()
                all_detections.extend(chunk_results)
                chunk_idx = futures[future]
                pbar.update(len(chunks[chunk_idx][0]))

    logger.info("Detected faces in %d images, running embeddings...", len(all_detections))

    # Phase 2: Batch embedding on GPU + matching
    matches = 0
    for batch_start in range(0, len(all_detections), config.batch_size):
        batch = all_detections[batch_start : batch_start + config.batch_size]

        all_crops = []
        crop_to_detection = []  # maps each crop to its index in batch
        for det_idx, (_, faces) in enumerate(batch):
            for face_idx in range(faces.shape[0]):
                all_crops.append(faces[face_idx])
                crop_to_detection.append(det_idx)

        if not all_crops:
            continue

        crops_tensor = torch.stack(all_crops).to(device)
        with torch.no_grad():
            embeddings = embedder(crops_tensor)

        matched = set()
        for crop_idx, det_idx in enumerate(crop_to_detection):
            if det_idx in matched:
                continue
            emb = embeddings[crop_idx].unsqueeze(0)
            sims = F.cosine_similarity(
                emb.unsqueeze(1),
                training_embeddings.unsqueeze(0),
                dim=2,
            )
            if sims.max().item() >= config.threshold:
                matched.add(det_idx)

        for det_idx in matched:
            src_path = Path(batch[det_idx][0])
            dest = copy_file(src_path, config.results_dir)
            logger.info("Match: %s → %s", src_path.name, dest.name)
            matches += 1

    return matches
