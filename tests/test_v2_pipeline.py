"""
Tests for V2 storm bounding box pipeline.
"""

import csv
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# =============================================================================
# STORM DETECTION — detect_largest_storm_bbox()
# =============================================================================

class TestDetectLargestStormBbox:

    def _make_temp_array_with_cold_blob(self, size=500, blob_center=(250, 250),
                                         blob_radius=50):
        """Create synthetic temp array with a cold blob (< -50°C)."""
        temp = np.full((size, size), 10.0, dtype=np.float32)  # warm background
        y, x = np.ogrid[:size, :size]
        mask = ((x - blob_center[0])**2 + (y - blob_center[1])**2) < blob_radius**2
        temp[mask] = -70.0  # cold storm
        return temp

    def test_returns_dict_with_bbox(self):
        from applyADT import detect_largest_storm_bbox
        temp = self._make_temp_array_with_cold_blob()
        result = detect_largest_storm_bbox(temp)
        assert result is not None
        assert 'bbox' in result
        assert len(result['bbox']) == 4
        minX, minY, maxX, maxY = result['bbox']
        assert minX < maxX
        assert minY < maxY

    def test_returns_none_when_no_cold_clouds(self):
        from applyADT import detect_largest_storm_bbox
        temp = np.full((500, 500), 10.0, dtype=np.float32)  # all warm
        result = detect_largest_storm_bbox(temp)
        assert result is None

    def test_returns_none_when_blobs_too_small(self):
        from applyADT import detect_largest_storm_bbox
        temp = np.full((500, 500), 10.0, dtype=np.float32)
        # Tiny cold spot (area < 500)
        temp[100:105, 100:105] = -70.0  # 25 pixels
        result = detect_largest_storm_bbox(temp, min_area=500)
        assert result is None

    def test_picks_largest_contour(self):
        from applyADT import detect_largest_storm_bbox
        temp = np.full((500, 500), 10.0, dtype=np.float32)
        # Small blob at (100, 100), radius 20
        y, x = np.ogrid[:500, :500]
        small = ((x - 100)**2 + (y - 100)**2) < 20**2
        temp[small] = -70.0
        # Large blob at (350, 350), radius 60
        large = ((x - 350)**2 + (y - 350)**2) < 60**2
        temp[large] = -70.0

        result = detect_largest_storm_bbox(temp)
        assert result is not None
        minX, minY, maxX, maxY = result['bbox']
        # Bbox center should be near the large blob (350, 350)
        cx = (minX + maxX) / 2
        cy = (minY + maxY) / 2
        assert abs(cx - 350) < 30
        assert abs(cy - 350) < 30

    def test_returns_area_and_center(self):
        from applyADT import detect_largest_storm_bbox
        temp = self._make_temp_array_with_cold_blob()
        result = detect_largest_storm_bbox(temp)
        assert 'area' in result
        assert result['area'] > 500
        assert 'center' in result
        cx, cy = result['center']
        assert abs(cx - 250) < 30
        assert abs(cy - 250) < 30


# =============================================================================
# BBOX AUGMENTATION — coordinate transforms
# =============================================================================

class TestBboxAugmentation:

    def test_rot90_identity(self):
        from v2.dataset import _transform_bbox_rot90
        bbox = (0.1, 0.2, 0.6, 0.8)
        result = _transform_bbox_rot90(bbox, 0)
        assert np.allclose(result, bbox, atol=1e-6)

    def test_rot90_full_rotation(self):
        from v2.dataset import _transform_bbox_rot90
        bbox = (0.1, 0.2, 0.6, 0.8)
        result = _transform_bbox_rot90(bbox, 4)  # 360° = identity
        assert np.allclose(result, bbox, atol=1e-6)

    def test_hflip_double_is_identity(self):
        from v2.dataset import _transform_bbox_hflip
        bbox = (0.1, 0.2, 0.6, 0.8)
        flipped = _transform_bbox_hflip(bbox)
        result = _transform_bbox_hflip(flipped)
        assert np.allclose(result, bbox, atol=1e-6)

    def test_hflip_preserves_y(self):
        from v2.dataset import _transform_bbox_hflip
        bbox = (0.1, 0.2, 0.6, 0.8)
        result = _transform_bbox_hflip(bbox)
        assert result[1] == bbox[1]  # minY unchanged
        assert result[3] == bbox[3]  # maxY unchanged

    def test_rot90_preserves_box_area(self):
        from v2.dataset import _transform_bbox_rot90
        bbox = (0.1, 0.2, 0.6, 0.8)
        orig_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        for k in range(4):
            r = _transform_bbox_rot90(bbox, k)
            area = (r[2] - r[0]) * (r[3] - r[1])
            assert abs(area - orig_area) < 1e-6, f"Area changed for k={k}"


# =============================================================================
# DATASET — BboxDataset
# =============================================================================

class TestBboxDataset:

    @pytest.fixture
    def synthetic_bbox_data(self, tmp_path):
        """Create a minimal synthetic bbox dataset."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()

        n_samples = 40  # enough for 70/15/15 split
        rows = []
        for i in range(n_samples):
            img = np.random.randn(768, 768).astype(np.float32) * 30 - 40
            fname = f"scene_{i:04d}.npy"
            np.save(images_dir / fname, img)
            rows.append({
                'image_file': fname,
                'minX': round(np.random.uniform(0.1, 0.3), 4),
                'minY': round(np.random.uniform(0.1, 0.3), 4),
                'maxX': round(np.random.uniform(0.5, 0.8), 4),
                'maxY': round(np.random.uniform(0.5, 0.8), 4),
                'area': np.random.randint(1000, 50000),
                'center_x': 1100,
                'center_y': 1100,
                'orig_h': 2200,
                'orig_w': 2200,
            })

        csv_path = tmp_path / "labels_bbox.csv"
        fieldnames = ['image_file', 'minX', 'minY', 'maxX', 'maxY', 'area',
                      'center_x', 'center_y', 'orig_h', 'orig_w']
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        return str(tmp_path)

    def test_dataset_returns_correct_shapes(self, synthetic_bbox_data):
        from v2.dataset import BboxDataset
        ds = BboxDataset(synthetic_bbox_data, split='train')
        img, bbox = ds[0]
        assert img.shape == (1, 768, 768)
        assert bbox.shape == (4,)

    def test_bbox_values_in_0_1(self, synthetic_bbox_data):
        from v2.dataset import BboxDataset
        ds = BboxDataset(synthetic_bbox_data, split='train')
        for i in range(min(5, len(ds))):
            _, bbox = ds[i]
            assert (bbox >= 0).all() and (bbox <= 1).all()

    def test_splits_are_disjoint(self, synthetic_bbox_data):
        from v2.dataset import BboxDataset
        train = BboxDataset(synthetic_bbox_data, split='train')
        val = BboxDataset(synthetic_bbox_data, split='val')
        test = BboxDataset(synthetic_bbox_data, split='test')
        train_set = set(train.split_indices)
        val_set = set(val.split_indices)
        test_set = set(test.split_indices)
        assert train_set.isdisjoint(val_set)
        assert train_set.isdisjoint(test_set)
        assert val_set.isdisjoint(test_set)

    def test_dataloaders_return_batches(self, synthetic_bbox_data):
        from v2.dataset import create_bbox_dataloaders
        train_loader, val_loader, test_loader = create_bbox_dataloaders(
            synthetic_bbox_data, batch_size=4, num_workers=0, augment=False)
        batch_x, batch_y = next(iter(train_loader))
        assert batch_x.shape == (4, 1, 768, 768)
        assert batch_y.shape == (4, 4)


# =============================================================================
# MODEL — StormBboxNet
# =============================================================================

class TestStormBboxNet:

    def test_output_shape(self):
        from v2.models import StormBboxNet
        model = StormBboxNet(pretrained=False)
        x = torch.randn(2, 1, 768, 768)
        out = model(x)
        assert out.shape == (2, 4)

    def test_output_in_0_1(self):
        from v2.models import StormBboxNet
        model = StormBboxNet(pretrained=False)
        model.eval()
        x = torch.randn(4, 1, 768, 768)
        with torch.no_grad():
            out = model(x)
        assert (out >= 0).all() and (out <= 1).all()

    def test_freeze_backbone(self):
        from v2.models import StormBboxNet
        model = StormBboxNet(pretrained=False)
        model.freeze_backbone()
        # Head params should be trainable
        head_trainable = any(p.requires_grad for p in model.head.parameters())
        assert head_trainable
        # conv1 should be frozen
        assert not model.conv1.weight.requires_grad

    def test_unfreeze_top_blocks(self):
        from v2.models import StormBboxNet
        model = StormBboxNet(pretrained=False)
        model.freeze_backbone()
        model.unfreeze_top_blocks()
        # layer3 should be trainable now
        layer3_trainable = any(p.requires_grad for p in model.layer3.parameters())
        assert layer3_trainable
        # layer1 should still be frozen
        layer1_frozen = all(not p.requires_grad for p in model.layer1.parameters())
        assert layer1_frozen

    def test_unfreeze_all(self):
        from v2.models import StormBboxNet
        model = StormBboxNet(pretrained=False)
        model.freeze_backbone()
        model.unfreeze_all()
        all_trainable = all(p.requires_grad for p in model.parameters())
        assert all_trainable

    def test_pretrained_1channel_adapter(self):
        from v2.models import StormBboxNet
        model = StormBboxNet(pretrained=True)
        assert model.conv1.weight.shape == (64, 1, 7, 7)


# =============================================================================
# IoU COMPUTATION
# =============================================================================

class TestComputeIoU:

    def test_perfect_overlap(self):
        from v2.evaluate import compute_iou
        pred = np.array([[0.1, 0.1, 0.5, 0.5]])
        target = np.array([[0.1, 0.1, 0.5, 0.5]])
        iou = compute_iou(pred, target)
        assert np.allclose(iou, 1.0)

    def test_no_overlap(self):
        from v2.evaluate import compute_iou
        pred = np.array([[0.0, 0.0, 0.2, 0.2]])
        target = np.array([[0.5, 0.5, 0.8, 0.8]])
        iou = compute_iou(pred, target)
        assert np.allclose(iou, 0.0)

    def test_partial_overlap(self):
        from v2.evaluate import compute_iou
        pred = np.array([[0.0, 0.0, 0.5, 0.5]])
        target = np.array([[0.25, 0.25, 0.75, 0.75]])
        iou = compute_iou(pred, target)
        # Intersection: [0.25, 0.25] to [0.5, 0.5] = 0.25 * 0.25 = 0.0625
        # pred area = 0.25, target area = 0.25
        # union = 0.25 + 0.25 - 0.0625 = 0.4375
        # IoU = 0.0625 / 0.4375 ≈ 0.1429
        assert 0.14 < iou[0] < 0.15

    def test_batch_computation(self):
        from v2.evaluate import compute_iou
        pred = np.array([[0.1, 0.1, 0.5, 0.5], [0.0, 0.0, 1.0, 1.0]])
        target = np.array([[0.1, 0.1, 0.5, 0.5], [0.0, 0.0, 1.0, 1.0]])
        iou = compute_iou(pred, target)
        assert iou.shape == (2,)
        assert np.allclose(iou, [1.0, 1.0])


# =============================================================================
# GIoU LOSS
# =============================================================================

class TestGIoULoss:

    def test_perfect_overlap_loss_zero(self):
        from v2.train import giou_loss
        pred = torch.tensor([[0.1, 0.1, 0.5, 0.5]])
        target = torch.tensor([[0.1, 0.1, 0.5, 0.5]])
        loss = giou_loss(pred, target)
        assert loss.item() < 0.01  # GIoU=1 → loss=0

    def test_no_overlap_loss_high(self):
        from v2.train import giou_loss
        pred = torch.tensor([[0.0, 0.0, 0.1, 0.1]])
        target = torch.tensor([[0.8, 0.8, 0.9, 0.9]])
        loss = giou_loss(pred, target)
        assert loss.item() > 1.0  # GIoU < 0 for distant boxes

    def test_giou_loss_trains_with_model(self, tmp_path):
        from v2.models import StormBboxNet
        from v2.train import train_bbox_model
        model = StormBboxNet(pretrained=False)
        dataset = torch.utils.data.TensorDataset(
            torch.randn(16, 1, 768, 768),
            torch.rand(16, 4).sort(dim=1).values,  # ensure minX<maxX, minY<maxY
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)

        class Args:
            checkpoint_dir = str(tmp_path / "ckpt_giou")
            epochs = 2
            lr = 1e-3
            loss = 'giou'
            resume = False

        history = train_bbox_model(model, loader, loader, Args())
        assert len(history['train_losses']) == 2


# =============================================================================
# DIoU LOSS
# =============================================================================

class TestDIoULoss:

    def test_perfect_overlap_loss_zero(self):
        from v2.train import diou_loss
        pred = torch.tensor([[0.1, 0.1, 0.5, 0.5]])
        target = torch.tensor([[0.1, 0.1, 0.5, 0.5]])
        loss = diou_loss(pred, target)
        assert loss.item() < 0.01  # IoU=1, center_dist=0 → DIoU=1 → loss=0

    def test_center_offset_penalized(self):
        """Two boxes with identical size but offset centers should produce
        a larger DIoU loss than two perfectly aligned boxes."""
        from v2.train import diou_loss
        # Offset prediction (shifted right by 0.1) — still partial overlap
        pred_offset = torch.tensor([[0.2, 0.1, 0.6, 0.5]])
        target = torch.tensor([[0.1, 0.1, 0.5, 0.5]])
        pred_aligned = torch.tensor([[0.1, 0.1, 0.5, 0.5]])
        loss_offset = diou_loss(pred_offset, target)
        loss_aligned = diou_loss(pred_aligned, target)
        assert loss_offset.item() > loss_aligned.item() + 0.1

    def test_diou_loss_trains_with_model(self, tmp_path):
        from v2.models import StormBboxNet
        from v2.train import train_bbox_model
        model = StormBboxNet(pretrained=False)
        dataset = torch.utils.data.TensorDataset(
            torch.randn(16, 1, 768, 768),
            torch.rand(16, 4).sort(dim=1).values,
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)

        class Args:
            checkpoint_dir = str(tmp_path / "ckpt_diou")
            epochs = 2
            lr = 1e-3
            loss = 'diou'
            resume = False

        history = train_bbox_model(model, loader, loader, Args())
        assert len(history['train_losses']) == 2


# =============================================================================
# MIXUP SMOKE TEST
# =============================================================================

class TestMixup:

    def test_mixup_runs_without_crash(self, tmp_path):
        from v2.models import StormBboxNet
        from v2.train import train_bbox_model
        model = StormBboxNet(pretrained=False)
        dataset = torch.utils.data.TensorDataset(
            torch.randn(16, 1, 768, 768),
            torch.rand(16, 4).sort(dim=1).values,
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)

        class Args:
            checkpoint_dir = str(tmp_path / "ckpt_mix")
            epochs = 2
            lr = 1e-3
            loss = 'smoothl1'
            resume = False
            mixup_alpha = 0.4

        history = train_bbox_model(model, loader, loader, Args())
        assert len(history['train_losses']) == 2


# =============================================================================
# TTA (Test-Time Augmentation)
# =============================================================================

class TestTTA:

    def test_identity_model_matches_baseline(self):
        """Constant-output model should yield the same bbox under TTA
        as a single forward pass, after inverse-rotating back."""
        from v2.evaluate import tta_predict

        class ConstModel(nn.Module):
            def __init__(self, bbox):
                super().__init__()
                self.register_buffer('bbox', torch.tensor(bbox).float())
                self.dummy = nn.Parameter(torch.zeros(1))

            def forward(self, x):
                return self.bbox.unsqueeze(0).expand(x.size(0), -1)

        # A centered symmetric bbox is invariant under 90° rotation,
        # so TTA average should match the baseline prediction.
        bbox = [0.3, 0.3, 0.7, 0.7]
        model = ConstModel(bbox).eval()
        x = torch.randn(2, 1, 16, 16)
        pred = tta_predict(model, x)
        assert np.allclose(pred[0], bbox, atol=1e-5)
        assert np.allclose(pred[1], bbox, atol=1e-5)

    def test_tta_output_shape(self):
        from v2.evaluate import tta_predict
        from v2.models import StormBboxNet

        model = StormBboxNet(pretrained=False).eval()
        x = torch.randn(3, 1, 768, 768)
        pred = tta_predict(model, x)
        assert pred.shape == (3, 4)
        assert (pred >= 0).all() and (pred <= 1).all()

    def test_tta_hflip_output_shape(self):
        from v2.evaluate import tta_predict
        from v2.models import StormBboxNet

        model = StormBboxNet(pretrained=False).eval()
        x = torch.randn(2, 1, 768, 768)
        pred = tta_predict(model, x, use_hflip=True)
        assert pred.shape == (2, 4)

    def test_inverse_rotation_recovers_bbox(self):
        """Forward-rotate a bbox by k, then inverse-rotate by (4-k)%4 — should recover."""
        from v2.dataset import _transform_bbox_rot90
        bbox = (0.1, 0.2, 0.6, 0.8)
        for k in range(4):
            rotated = _transform_bbox_rot90(bbox, k)
            recovered = _transform_bbox_rot90(rotated, (4 - k) % 4)
            assert np.allclose(recovered, bbox, atol=1e-6), f"k={k} failed"


# =============================================================================
# TRAINING LOOP SMOKE TEST
# =============================================================================

class TestTrainingLoop:

    def test_train_runs_without_crash(self, tmp_path):
        """Smoke test: training loop runs 2 epochs on synthetic data."""
        from v2.models import StormBboxNet
        from v2.train import train_bbox_model

        model = StormBboxNet(pretrained=False)

        # Synthetic dataloaders
        dataset = torch.utils.data.TensorDataset(
            torch.randn(16, 1, 768, 768),
            torch.rand(16, 4),
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)

        class Args:
            checkpoint_dir = str(tmp_path / "ckpt")
            epochs = 2
            lr = 1e-3
            resume = False

        history = train_bbox_model(model, loader, loader, Args())
        assert len(history['train_losses']) == 2
        assert len(history['val_losses']) == 2

    def test_checkpoint_resume(self, tmp_path):
        """Train 2 epochs, resume, train 2 more."""
        from v2.models import StormBboxNet
        from v2.train import train_bbox_model

        model = StormBboxNet(pretrained=False)
        dataset = torch.utils.data.TensorDataset(
            torch.randn(16, 1, 768, 768),
            torch.rand(16, 4),
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)

        ckpt_dir = str(tmp_path / "ckpt")

        class Args:
            checkpoint_dir = ckpt_dir
            epochs = 4
            lr = 1e-3
            resume = False

        # Train 2 epochs
        Args.epochs = 2
        train_bbox_model(model, loader, loader, Args())
        assert os.path.exists(os.path.join(ckpt_dir, 'latest.pt'))

        # Resume and train 2 more
        Args.resume = True
        Args.epochs = 4
        model2 = StormBboxNet(pretrained=False)
        history = train_bbox_model(model2, loader, loader, Args())
        assert len(history['train_losses']) == 4
