"""
Batch ADT module - generate dataset from PNG/CBOR pairs.

Two-phase design for resilience:
  Phase 1: Process pairs in parallel → .npy patch + .json sidecar per patch (resumable)
  Phase 2: Scan .json sidecars → write labels.csv (idempotent)
"""

import csv
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np

from applyADT import auto_detect_storm_and_apply_adt, load_temperature_data_from_png

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LABEL_FIELDS = ['patch_file', 't_number', 'wind_knots', 'pressure_hpa',
                'eye_temp', 'eyewall_temp', 'delta_t', 'center_x', 'center_y']


def _validate_existing_patch(patch_path):
    """Check if an existing .npy patch is valid (correct shape and dtype)."""
    try:
        patch = np.load(patch_path)
        return patch.shape == (240, 240) and patch.dtype == np.float32
    except Exception:
        return False


def process_single_pair(png_path, cbor_path, patches_dir):
    """Process one PNG/CBOR pair → save .npy patch + .json sidecar.

    Args:
        png_path: Path to GK2A_IR105 PNG file
        cbor_path: Path to product.cbor calibration file
        patches_dir: Directory to save patches and sidecars

    Returns:
        (n_patches, skipped, error_msg) where error_msg is None on success
    """
    scene_id = Path(png_path).stem

    # Check if already processed (resume support)
    existing = list(Path(patches_dir).glob(f"{scene_id}_*.npy"))
    if existing and all(_validate_existing_patch(p) for p in existing):
        return (0, 0, None)  # already done, counted as neither new nor skipped

    try:
        temp_c = load_temperature_data_from_png(png_path, cbor_path)
        results = auto_detect_storm_and_apply_adt(temp_c)

        if isinstance(results, dict) and 'error' in results:
            return (0, 1, None)
        if not results:
            return (0, 1, None)

        n_patches = 0
        for storm_info in results:
            storm_id = storm_info.get('Storm_ID', 0)
            patch_filename = f"{scene_id}_{storm_id}.npy"
            patch_path = os.path.join(patches_dir, patch_filename)
            sidecar_path = os.path.join(patches_dir, f"{scene_id}_{storm_id}.json")

            center = storm_info.get('Detected_Center')
            if center is None:
                continue
            center_x, center_y = center

            # Extract 240x240 patch centered at storm
            half_box = 120
            h, w = temp_c.shape
            y1 = max(0, center_y - half_box)
            y2 = min(h, center_y + half_box)
            x1 = max(0, center_x - half_box)
            x2 = min(w, center_x + half_box)

            patch = temp_c[y1:y2, x1:x2]

            # Pad if needed
            if patch.shape[0] < 240 or patch.shape[1] < 240:
                padded = np.zeros((240, 240), dtype=np.float32)
                padded[:patch.shape[0], :patch.shape[1]] = patch
                patch = padded
            elif patch.shape[0] > 240 or patch.shape[1] > 240:
                patch = patch[:240, :240]

            patch = patch.astype(np.float32)
            np.save(patch_path, patch)

            # Write sidecar JSON (atomic unit of metadata)
            sidecar = {
                'patch_file': patch_filename,
                't_number': storm_info.get('T-number', 0),
                'wind_knots': storm_info.get('Wind_Speed_knots', 0),
                'pressure_hpa': storm_info.get('Pressure_hPa', 0),
                'eye_temp': storm_info.get('Eye_Temp_C', 0),
                'eyewall_temp': storm_info.get('Eyewall_Temp_C', 0),
                'delta_t': storm_info.get('Delta_T', 0),
                'center_x': center_x,
                'center_y': center_y,
            }
            with open(sidecar_path, 'w') as f:
                json.dump(sidecar, f)

            n_patches += 1

        return (n_patches, 0, None)

    except Exception as e:
        return (0, 0, str(e))


def collect_labels(patches_dir, output_csv):
    """Scan .json sidecars in patches_dir → write labels.csv. Idempotent.

    Args:
        patches_dir: Directory containing .json sidecar files
        output_csv: Path to write labels.csv

    Returns:
        Number of rows written
    """
    sidecars = sorted(Path(patches_dir).glob("*.json"))

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
    """Generate dataset from PNG/CBOR pairs with concurrency and resume support.

    Phase 1: Process pairs in parallel, saving .npy + .json sidecar per patch.
             Skips pairs whose patches already exist (resume).
    Phase 2: Collect all .json sidecars → write labels.csv.

    Args:
        pairs: list of (png_path, cbor_path) tuples
        output_dir: Directory to save patches/ and labels.csv
        workers: Number of parallel workers (default: min(cpu_count, 4))

    Returns:
        dict with total_patches, skipped_scenes, errors, resumed counts
    """
    if workers is None:
        workers = min(cpu_count(), 4)

    patches_dir = os.path.join(output_dir, "patches")
    os.makedirs(patches_dir, exist_ok=True)

    total_patches = 0
    skipped_scenes = 0
    errors = 0
    resumed = 0

    if not pairs:
        csv_path = os.path.join(output_dir, "labels.csv")
        collect_labels(patches_dir, csv_path)
        return {
            'total_patches': 0,
            'skipped_scenes': 0,
            'errors': 0,
            'resumed': 0,
        }

    # Phase 1: parallel patch generation
    use_tqdm = tqdm is not None
    iterator = pairs

    if workers <= 1:
        # Sequential mode (for testing or single-core)
        if use_tqdm:
            iterator = tqdm(pairs, desc="Processing pairs", unit="pair")

        for png_path, cbor_path in iterator:
            n_patches, skipped, error = process_single_pair(png_path, cbor_path, patches_dir)
            if error:
                errors += 1
                logger.warning(f"Error processing {Path(png_path).stem}: {error}")
            elif n_patches == 0 and skipped == 0:
                resumed += 1
            else:
                total_patches += n_patches
                skipped_scenes += skipped

            if use_tqdm:
                iterator.set_postfix(patches=total_patches, skipped=skipped_scenes,
                                     errors=errors, resumed=resumed)
    else:
        # Parallel mode
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_single_pair, png, cbor, patches_dir): (png, cbor)
                for png, cbor in pairs
            }

            completed_iter = as_completed(futures)
            if use_tqdm:
                completed_iter = tqdm(completed_iter, total=len(futures),
                                      desc="Processing pairs", unit="pair")

            for future in completed_iter:
                png_path, _ = futures[future]
                try:
                    n_patches, skipped, error = future.result()
                    if error:
                        errors += 1
                        logger.warning(f"Error processing {Path(png_path).stem}: {error}")
                    elif n_patches == 0 and skipped == 0:
                        resumed += 1
                    else:
                        total_patches += n_patches
                        skipped_scenes += skipped
                except Exception as e:
                    errors += 1
                    logger.warning(f"Worker exception for {Path(png_path).stem}: {e}")

                if use_tqdm:
                    completed_iter.set_postfix(patches=total_patches, skipped=skipped_scenes,
                                               errors=errors, resumed=resumed)

    # Phase 2: collect sidecars → labels.csv
    csv_path = os.path.join(output_dir, "labels.csv")
    csv_rows = collect_labels(patches_dir, csv_path)
    logger.info(f"labels.csv written: {csv_rows} rows")

    summary = {
        'total_patches': total_patches,
        'skipped_scenes': skipped_scenes,
        'errors': errors,
        'resumed': resumed,
    }
    logger.info(f"Done: {total_patches} patches, {skipped_scenes} no-storm, "
                f"{errors} errors, {resumed} resumed")
    return summary
