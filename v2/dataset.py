"""
Dataset for storm bounding box regression.
Loads 512x512 resized full-disk IR images + normalized bbox targets.
"""

import csv
import os

import numpy as np
import torch
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


def _transform_bbox_rot90(bbox, k):
    """Transform normalized bbox (minX, minY, maxX, maxY) for k*90° rotation.

    Applies the same coordinate transform as np.rot90(image, k).
    All coords in [0, 1].
    """
    minX, minY, maxX, maxY = bbox
    for _ in range(k % 4):
        # 90° CCW: (x, y) → (y, 1-x)
        minX, minY, maxX, maxY = minY, 1 - maxX, maxY, 1 - minX
    return (minX, minY, maxX, maxY)


def _transform_bbox_hflip(bbox):
    """Transform normalized bbox for horizontal flip."""
    minX, minY, maxX, maxY = bbox
    return (1 - maxX, minY, 1 - minX, maxY)


class BboxDataset(Dataset):
    """Full-disk image dataset for storm bounding box regression."""

    def __init__(self, data_dir, split='train', seed=42, augment=False):
        self.data_dir = data_dir
        self.split = split
        self.augment = augment

        labels_path = os.path.join(data_dir, "labels_bbox.csv")
        with open(labels_path, 'r') as f:
            reader = csv.DictReader(f)
            self.labels = list(reader)

        # Stratify by bbox area bin (small/medium/large/xlarge)
        areas = [float(row['area']) for row in self.labels]
        area_bins = []
        q25, q50, q75 = np.percentile(areas, [25, 50, 75])
        for a in areas:
            if a < q25:
                area_bins.append(0)
            elif a < q50:
                area_bins.append(1)
            elif a < q75:
                area_bins.append(2)
            else:
                area_bins.append(3)

        indices = list(range(len(self.labels)))

        # 70/15/15 split
        train_idx, temp_idx = train_test_split(
            indices, test_size=0.30, random_state=seed, stratify=area_bins
        )

        temp_bins = [area_bins[i] for i in temp_idx]
        bin_counts = Counter(temp_bins)
        can_stratify = all(count >= 2 for count in bin_counts.values())

        if can_stratify:
            val_idx, test_idx = train_test_split(
                temp_idx, test_size=0.50, random_state=seed, stratify=temp_bins
            )
        else:
            val_idx, test_idx = train_test_split(
                temp_idx, test_size=0.50, random_state=seed
            )

        if split == 'train':
            self.split_indices = train_idx
        elif split == 'val':
            self.split_indices = val_idx
        elif split == 'test':
            self.split_indices = test_idx
        else:
            raise ValueError(f"Unknown split: {split}")

    def __len__(self):
        return len(self.split_indices)

    def __getitem__(self, idx):
        """Returns (image_tensor [1, 512, 512], bbox_tensor [4])."""
        real_idx = self.split_indices[idx]
        row = self.labels[real_idx]

        image_path = os.path.join(self.data_dir, "images", row['image_file'])
        image = np.load(image_path).astype(np.float32)

        # Normalize: (temp + 90) / 130 — same as V1
        image = (image + 90.0) / 130.0

        bbox = (
            float(row['minX']),
            float(row['minY']),
            float(row['maxX']),
            float(row['maxY']),
        )

        if self.augment:
            k = np.random.randint(0, 4)
            image = np.rot90(image, k).copy()
            bbox = _transform_bbox_rot90(bbox, k)

            if np.random.random() > 0.5:
                image = np.fliplr(image).copy()
                bbox = _transform_bbox_hflip(bbox)

        image_tensor = torch.from_numpy(image).unsqueeze(0).float()
        bbox_tensor = torch.tensor(bbox, dtype=torch.float32)

        return image_tensor, bbox_tensor


def create_bbox_dataloaders(data_dir, batch_size=16, seed=42, augment=True,
                            num_workers=4):
    """Create train/val/test dataloaders for bbox regression."""
    train_dataset = BboxDataset(data_dir, split='train', seed=seed, augment=augment)
    val_dataset = BboxDataset(data_dir, split='val', seed=seed)
    test_dataset = BboxDataset(data_dir, split='test', seed=seed)

    pin = num_workers > 0
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin,
        persistent_workers=pin,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
        persistent_workers=pin,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
        persistent_workers=pin,
    )

    return train_loader, val_loader, test_loader
