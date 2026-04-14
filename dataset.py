"""
Dataset module for typhoon intensity estimation.
"""

import csv
import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from collections import Counter
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class TyphoonDataset(Dataset):
    """Typhoon dataset for intensity estimation."""

    def __init__(self, data_dir, split='train', seed=42, augment=False):
        """
        Initialize dataset with data directory and split.

        Args:
            data_dir: Directory containing labels.csv and patches/
            split: 'train', 'val', or 'test'
            seed: Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.split = split
        self.augment = augment

        # Read labels.csv
        labels_path = os.path.join(data_dir, "labels.csv")
        with open(labels_path, 'r') as f:
            reader = csv.DictReader(f)
            self.labels = list(reader)

        # Create indices for stratification
        # Bin T-numbers for stratification: 1.0-2.5, 2.5-4.0, 4.0-5.5, 5.5+
        t_numbers = [float(row['t_number']) for row in self.labels]
        t_bins = []
        for t in t_numbers:
            if t < 2.5:
                t_bins.append(0)
            elif t < 4.0:
                t_bins.append(1)
            elif t < 5.5:
                t_bins.append(2)
            else:
                t_bins.append(3)

        # Split: 70% train, 15% val, 15% test
        indices = list(range(len(self.labels)))

        # First split: 70% train, 30% temp
        train_idx, temp_idx = train_test_split(
            indices,
            test_size=0.30,
            random_state=seed,
            stratify=t_bins
        )

        # Second split: split temp into 50/50 for val/test (which gives 15% each)
        temp_labels = [t_bins[i] for i in temp_idx]

        # If all bins have at least 2 samples, we can stratify
        from collections import Counter
        temp_bin_counts = Counter(temp_labels)
        can_stratify = all(count >= 2 for count in temp_bin_counts.values())

        if can_stratify:
            val_idx, test_idx = train_test_split(
                temp_idx,
                test_size=0.50,
                random_state=seed,
                stratify=temp_labels
            )
        else:
            # Can't stratify - fall back to random split
            val_idx, test_idx = train_test_split(
                temp_idx,
                test_size=0.50,
                random_state=seed
            )

        # Map split name to indices
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
        """
        Get item by index.

        Returns:
            patch_tensor: [1, 240, 240] normalized temperature
            targets: [3] tensor of [t_number, wind_knots, pressure_hpa]
        """
        real_idx = self.split_indices[idx]
        row = self.labels[real_idx]

        # Load patch
        patch_file = row['patch_file']
        patch_path = os.path.join(self.data_dir, "patches", patch_file)
        patch = np.load(patch_path).astype(np.float32)

        # Normalize: (temp - (-90)) / (40 - (-90)) = (temp + 90) / 130
        patch = (patch + 90.0) / 130.0

        # Augmentation: random rotation (0/90/180/270) + random flip
        # Valid for satellite IR — typhoon structure has no preferred orientation
        if self.augment:
            k = np.random.randint(0, 4)
            patch = np.rot90(patch, k).copy()
            if np.random.random() > 0.5:
                patch = np.fliplr(patch).copy()

        # Convert to tensor [1, 240, 240]
        patch_tensor = torch.from_numpy(patch).unsqueeze(0).float()

        # Get targets
        t_number = float(row['t_number'])
        wind_knots = float(row['wind_knots'])
        pressure_hpa = float(row['pressure_hpa'])
        targets = torch.tensor([t_number, wind_knots, pressure_hpa], dtype=torch.float32)

        return patch_tensor, targets


def create_dataloaders(data_dir, batch_size=32, seed=42):
    """
    Create train, val, test dataloaders.

    Args:
        data_dir: Directory containing labels.csv and patches/
        batch_size: Batch size for dataloaders
        seed: Random seed for reproducibility

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_dataset = TyphoonDataset(data_dir, split='train', seed=seed)
    val_dataset = TyphoonDataset(data_dir, split='val', seed=seed)
    test_dataset = TyphoonDataset(data_dir, split='test', seed=seed)

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=g
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, test_loader


def _compute_bin_weights(dataset):
    """Compute inverse-frequency sampling weights by T-number bin.

    Bins: [1.0, 2.5), [2.5, 4.0), [4.0, 5.5), [5.5, 7.0), [7.0, 8.01)
    Weight per sample = 1/count_of_its_bin, normalized so mean weight = 1.
    """
    t_numbers = []
    for idx in dataset.split_indices:
        t_numbers.append(float(dataset.labels[idx]['t_number']))

    bin_edges = [1.0, 2.5, 4.0, 5.5, 7.0, 8.01]
    sample_bins = []
    for t in t_numbers:
        for j in range(len(bin_edges) - 1):
            if t < bin_edges[j + 1]:
                sample_bins.append(j)
                break
        else:
            sample_bins.append(len(bin_edges) - 2)

    bin_counts = Counter(sample_bins)
    raw_weights = [1.0 / bin_counts[b] for b in sample_bins]
    mean_w = sum(raw_weights) / len(raw_weights)
    weights = [w / mean_w for w in raw_weights]
    return torch.tensor(weights, dtype=torch.float64)


def create_balanced_dataloaders(data_dir, batch_size=32, seed=42, augment=False,
                                num_workers=4):
    """Create dataloaders with inverse-frequency weighted sampling for training.

    Train loader uses WeightedRandomSampler so all T-number bins contribute
    equal expected gradients. Val and test loaders are unweighted.
    """
    train_dataset = TyphoonDataset(data_dir, split='train', seed=seed, augment=augment)
    val_dataset = TyphoonDataset(data_dir, split='val', seed=seed)
    test_dataset = TyphoonDataset(data_dir, split='test', seed=seed)

    weights = _compute_bin_weights(train_dataset)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    pin = num_workers > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=pin,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=pin,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=pin,
    )

    return train_loader, val_loader, test_loader