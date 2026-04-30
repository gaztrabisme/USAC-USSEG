"""
Batch bounding box label generation from PNG/CBOR pairs.

Two-phase sidecar pattern (same resilience as batch_adt.py):
  Phase 1: Process pairs in parallel → resized .npy + .json sidecar (resumable)
  Phase 2: Scan .json sidecars → write labels_bbox.csv (idempotent)
"""

import csv
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from applyADT import detect_largest_storm_bbox, load_temperature_data_from_png

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LABEL_FIELDS = ['image_file', 'minX', 'minY', 'maxX', 'maxY', 'area',
                'center_x', 'center_y', 'orig_h', 'orig_w']
TARGET_SIZE = 768


def _validate_existing_image(npy_path):
    """Check if an existing .npy resized image is valid."""
    try:
        img = np.load(npy_path)
        return img.shape == (TARGET_SIZE, TARGET_SIZE) and img.dtype == np.float32
    except Exception:
        return False


def process_single_pair(png_path, cbor_path, images_dir):
    """Process one PNG/CBOR pair → resized .npy image + .json sidecar.

    Returns:
        (success, skipped, error_msg) where error_msg is None on success
    """
    scene_id = Path(png_path).stem
    npy_path = os.path.join(images_dir, f"{scene_id}.npy")
    sidecar_path = os.path.join(images_dir, f"{scene_id}.json")

    # Resume support
    if os.path.exists(npy_path) and os.path.exists(sidecar_path):
        if _validate_existing_image(npy_path):
            return (0, 0, None)  # already done

    try:
        temp_c = load_temperature_data_from_png(png_path, cbor_path)
        result = detect_largest_storm_bbox(temp_c)

        if result is None:
            return (0, 1, None)  # no storm found

        orig_h, orig_w = temp_c.shape
        minX, minY, maxX, maxY = result['bbox']
        cx, cy = result['center']

        # Resize full-disk to TARGET_SIZE x TARGET_SIZE
        resized = cv2.resize(temp_c.astype(np.float32),
                             (TARGET_SIZE, TARGET_SIZE),
                             interpolation=cv2.INTER_LINEAR)
        np.save(npy_path, resized)

        # Normalize bbox to [0, 1]
        sidecar = {
            'image_file': f"{scene_id}.npy",
            'minX': round(minX / orig_w, 6),
            'minY': round(minY / orig_h, 6),
            'maxX': round(maxX / orig_w, 6),
            'maxY': round(maxY / orig_h, 6),
            'area': result['area'],
            'center_x': cx,
            'center_y': cy,
            'orig_h': orig_h,
            'orig_w': orig_w,
        }
        with open(sidecar_path, 'w') as f:
            json.dump(sidecar, f)

        return (1, 0, None)

    except Exception as e:
        return (0, 0, str(e))


def collect_labels(images_dir, output_csv):
    """Scan .json sidecars → write labels_bbox.csv. Idempotent."""
    sidecars = sorted(Path(images_dir).glob("*.json"))
    rows = []
    for sidecar_path in sidecars:
        with open(sidecar_path) as f:
            rows.append(json.load(f))

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=LABEL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def generate_dataset(pairs, output_dir, workers=None):
    """Generate bbox dataset from PNG/CBOR pairs.

    Phase 1: Process pairs in parallel → resized .npy + .json sidecar.
    Phase 2: Collect sidecars → labels_bbox.csv.
    """
    if workers is None:
        workers = min(cpu_count(), 4)

    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    total = 0
    skipped = 0
    errors = 0
    resumed = 0

    if not pairs:
        csv_path = os.path.join(output_dir, "labels_bbox.csv")
        collect_labels(images_dir, csv_path)
        return {'total': 0, 'skipped': 0, 'errors': 0, 'resumed': 0}

    use_tqdm = tqdm is not None

    if workers <= 1:
        iterator = pairs
        if use_tqdm:
            iterator = tqdm(pairs, desc="Processing pairs", unit="pair")

        for png_path, cbor_path in iterator:
            success, skip, error = process_single_pair(png_path, cbor_path, images_dir)
            if error:
                errors += 1
                logger.warning(f"Error processing {Path(png_path).stem}: {error}")
            elif success == 0 and skip == 0:
                resumed += 1
            else:
                total += success
                skipped += skip

            if use_tqdm:
                iterator.set_postfix(total=total, skipped=skipped, errors=errors)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_single_pair, png, cbor, images_dir): (png, cbor)
                for png, cbor in pairs
            }

            completed_iter = as_completed(futures)
            if use_tqdm:
                completed_iter = tqdm(completed_iter, total=len(futures),
                                      desc="Processing pairs", unit="pair")

            for future in completed_iter:
                png_path, _ = futures[future]
                try:
                    success, skip, error = future.result()
                    if error:
                        errors += 1
                        logger.warning(f"Error: {Path(png_path).stem}: {error}")
                    elif success == 0 and skip == 0:
                        resumed += 1
                    else:
                        total += success
                        skipped += skip
                except Exception as e:
                    errors += 1
                    logger.warning(f"Worker exception: {Path(png_path).stem}: {e}")

                if use_tqdm:
                    completed_iter.set_postfix(total=total, skipped=skipped, errors=errors)

    csv_path = os.path.join(output_dir, "labels_bbox.csv")
    csv_rows = collect_labels(images_dir, csv_path)
    logger.info(f"labels_bbox.csv written: {csv_rows} rows")

    summary = {'total': total, 'skipped': skipped, 'errors': errors, 'resumed': resumed}
    logger.info(f"Done: {total} images, {skipped} no-storm, {errors} errors, {resumed} resumed")
    return summary
