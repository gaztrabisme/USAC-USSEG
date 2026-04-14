"""
Prepare data module - scan pre-extracted fragment directories for PNG+CBOR pairs.
"""

import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SKIP_DIRS = {"GK-2A_inference", "GOES-18_cleaned"}


def scan_and_pair(data_dir):
    """
    Scan pre-extracted fragment directories for GK2A_IR105_*.png + product.cbor pairs.

    Walks each fragment subfolder under data_dir, looking for timestamped
    directories that contain both a GK2A_IR105_*.png and a product.cbor file.

    Args:
        data_dir: Root directory containing fragment subdirectories
                  (e.g. 'projects/USAC/GK-2A/data')

    Returns:
        (pairs, orphans) where:
            pairs: list of (png_path, cbor_path) tuples
            orphans: list of strings describing unmatched files
    """
    pairs = []
    orphans = []

    if not os.path.isdir(data_dir):
        logger.error(f"Data directory does not exist: {data_dir}")
        return pairs, orphans

    for entry in sorted(os.listdir(data_dir)):
        if entry in SKIP_DIRS:
            logger.info(f"Skipping: {entry}")
            continue

        entry_path = os.path.join(data_dir, entry)
        if not os.path.isdir(entry_path):
            continue

        for root, _dirs, files in os.walk(entry_path):
            has_cbor = "product.cbor" in files
            ir_pngs = [f for f in files
                       if f.startswith("GK2A_IR105_") and f.endswith(".png")]

            if has_cbor and ir_pngs:
                png_path = os.path.join(root, ir_pngs[0])
                cbor_path = os.path.join(root, "product.cbor")
                pairs.append((png_path, cbor_path))
            elif has_cbor:
                orphans.append(os.path.join(root, "product.cbor"))
            elif ir_pngs:
                orphans.append(os.path.join(root, ir_pngs[0]))

    logger.info(f"Found {len(pairs)} pair(s), {len(orphans)} orphan(s)")
    return pairs, orphans
