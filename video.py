import gc
import logging
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from tqdm import tqdm

from config import Config, select_device

logger = logging.getLogger(__name__)

SAMPLE_FPS = 4.0
GAP_TOLERANCE = 0.5
MIN_SEGMENT_DURATION = 1.0
VIDEO_MAX_SIZE = 1080  # balance between memory and detecting smaller/distant faces


def run_video_detection(config: Config, training_embeddings: torch.Tensor) -> Path:
    """Scan a video for the target person and write time segments to a file."""
    video_path = config.video_path
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / native_fps if native_fps > 0 else 0
    logger.info("Video: %s (%.1f fps, %d frames, %.1fs)", video_path.name, native_fps, total_frames, duration)

    sample_interval = max(1, int(round(native_fps / SAMPLE_FPS)))
    logger.info("Sampling every %d frames (≈%.1f fps)", sample_interval, native_fps / sample_interval)

    # Determine frame range from start/end times
    start_frame = 0
    end_frame = total_frames
    if config.video_start is not None:
        start_frame = int(config.video_start * native_fps)
        logger.info("Starting at %s (frame %d)", _format_timestamp(config.video_start), start_frame)
    if config.video_end is not None:
        end_frame = min(total_frames, int(config.video_end * native_fps))
        logger.info("Ending at %s (frame %d)", _format_timestamp(config.video_end), end_frame)

    # Seek to start frame
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    device = select_device()
    mtcnn = MTCNN(keep_all=True, min_face_size=15, device=torch.device("cpu"), post_process=True)
    embedder = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    training_embeddings = training_embeddings.to(device)

    matched_timestamps: list[float] = []
    pending: list[tuple[float, torch.Tensor]] = []  # (timestamp, face_crops)

    frames_to_process = end_frame - start_frame
    frame_idx = start_frame
    with tqdm(total=frames_to_process, desc="Scanning video", unit="frame") as pbar:
        while frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_interval == 0:
                timestamp = frame_idx / native_fps
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                del rgb

                # Resize for face detection — VIDEO_MAX_SIZE is enough
                max_size = min(config.max_size, VIDEO_MAX_SIZE)
                w, h = pil_img.size
                longest = max(w, h)
                if longest > max_size:
                    scale = max_size / longest
                    pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

                faces = mtcnn(pil_img)
                del pil_img
                if faces is not None:
                    if faces.dim() == 3:
                        faces = faces.unsqueeze(0)
                    pending.append((timestamp, faces.cpu()))

                if len(pending) >= config.batch_size:
                    _flush_batch(pending, embedder, training_embeddings, config.threshold, matched_timestamps, device)
                    pending.clear()
                    gc.collect()
                    if device.type == "mps":
                        torch.mps.empty_cache()

            del frame
            frame_idx += 1
            pbar.update(1)

    cap.release()

    # Flush remaining
    if pending:
        _flush_batch(pending, embedder, training_embeddings, config.threshold, matched_timestamps, device)
        pending.clear()

    # Release model memory
    del mtcnn, embedder
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    logger.info("Found target person in %d sampled frames", len(matched_timestamps))

    sample_interval_sec = sample_interval / native_fps
    gap_tolerance = sample_interval_sec + GAP_TOLERANCE
    segments = _build_segments(matched_timestamps, gap_tolerance, MIN_SEGMENT_DURATION)
    logger.info("Built %d segments (min %.1fs, gap tolerance %.2fs)", len(segments), MIN_SEGMENT_DURATION, gap_tolerance)

    output_path = _get_output_path(video_path, config)
    _write_segments(segments, output_path)
    return output_path


def _flush_batch(
    pending: list[tuple[float, torch.Tensor]],
    embedder: InceptionResnetV1,
    training_embeddings: torch.Tensor,
    threshold: float,
    matched_timestamps: list[float],
    device: torch.device,
) -> None:
    """Embed accumulated face crops and check for matches."""
    all_crops = []
    crop_to_timestamp: list[float] = []

    for timestamp, faces in pending:
        for i in range(faces.shape[0]):
            all_crops.append(faces[i])
            crop_to_timestamp.append(timestamp)

    if not all_crops:
        return

    crops_tensor = torch.stack(all_crops).to(device)
    with torch.no_grad():
        embeddings = embedder(crops_tensor)

    matched_ts = set()
    for crop_idx, ts in enumerate(crop_to_timestamp):
        if ts in matched_ts:
            continue
        emb = embeddings[crop_idx].unsqueeze(0)
        sims = F.cosine_similarity(
            emb.unsqueeze(1),
            training_embeddings.unsqueeze(0),
            dim=2,
        )
        if sims.max().item() >= threshold:
            matched_ts.add(ts)

    for ts in sorted(matched_ts):
        matched_timestamps.append(ts)


def _build_segments(
    matched_timestamps: list[float],
    gap_tolerance: float,
    min_duration: float,
) -> list[tuple[float, float]]:
    """Group timestamps into segments, merge close ones, filter short ones."""
    if not matched_timestamps:
        return []

    timestamps = sorted(matched_timestamps)
    segments = []
    start = timestamps[0]
    end = timestamps[0]

    for ts in timestamps[1:]:
        if ts - end <= gap_tolerance:
            end = ts
        else:
            segments.append((start, end))
            start = ts
            end = ts
    segments.append((start, end))

    return [(s, e) for s, e in segments if e - s >= min_duration]


def _format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS."""
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _write_segments(segments: list[tuple[float, float]], output_path: Path) -> None:
    """Write segments to a text file."""
    with open(output_path, "w") as f:
        for start, end in segments:
            f.write(f"{_format_timestamp(start)} - {_format_timestamp(end)}\n")
    logger.info("Wrote %d segments to %s", len(segments), output_path)


def _get_output_path(video_path: Path, config: Config) -> Path:
    """Return output path: {video_stem}_segments.txt next to the video."""
    return video_path.parent / f"{video_path.stem}_segments.txt"
