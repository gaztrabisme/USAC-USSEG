"""
Evaluation for storm bounding box regression.

Metrics: mean IoU, IoU@0.5 accuracy, per-sample IoU.
Visualization: bbox overlays on best/worst predictions.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def compute_iou(pred, target):
    """Compute IoU between predicted and target bounding boxes.

    Args:
        pred: (N, 4) array of predicted boxes (minX, minY, maxX, maxY), normalized [0,1]
        target: (N, 4) array of target boxes

    Returns:
        (N,) array of IoU values
    """
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    # Intersection
    inter_minX = np.maximum(pred[:, 0], target[:, 0])
    inter_minY = np.maximum(pred[:, 1], target[:, 1])
    inter_maxX = np.minimum(pred[:, 2], target[:, 2])
    inter_maxY = np.minimum(pred[:, 3], target[:, 3])

    inter_w = np.maximum(0, inter_maxX - inter_minX)
    inter_h = np.maximum(0, inter_maxY - inter_minY)
    inter_area = inter_w * inter_h

    # Union
    pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union_area = pred_area + target_area - inter_area

    iou = np.where(union_area > 0, inter_area / union_area, 0.0)
    return iou


def tta_predict(model, batch_x, use_hflip=False):
    """Test-time augmentation: predict over 4 rotations (optionally + hflip),
    inverse-transform each prediction back to the original frame, average.

    Args:
        model: StormBboxNet in eval mode
        batch_x: [B, 1, H, W] tensor on model's device
        use_hflip: if True, also include horizontal-flip variants (8 total)

    Returns:
        [B, 4] numpy array of averaged predictions (minX, minY, maxX, maxY)
    """
    from v2.dataset import _transform_bbox_hflip, _transform_bbox_rot90

    preds_list = []
    flip_options = (False, True) if use_hflip else (False,)

    with torch.no_grad():
        for k in range(4):
            for do_flip in flip_options:
                img = torch.rot90(batch_x, k, dims=(-2, -1))
                if do_flip:
                    img = torch.flip(img, dims=(-1,))
                pred = model(img).cpu().numpy()  # [B, 4] in transformed frame

                recovered = np.empty_like(pred)
                for i, bbox in enumerate(pred):
                    b = _transform_bbox_hflip(tuple(bbox)) if do_flip else tuple(bbox)
                    b = _transform_bbox_rot90(b, (4 - k) % 4)
                    recovered[i] = b
                preds_list.append(recovered)

    return np.mean(np.stack(preds_list, axis=0), axis=0)


def evaluate_bbox_model(model, test_loader, checkpoint_path, tta=False,
                        tta_hflip=False):
    """Evaluate bbox model on test set.

    Args:
        model: StormBboxNet instance
        test_loader: Test data loader
        checkpoint_path: Path to best_model.pt
        tta: if True, apply test-time augmentation (4 rotations)
        tta_hflip: if True (and tta=True), also include hflip variants (8-way)

    Returns:
        (metrics, predictions, targets) where metrics is
        {mean_iou, iou_at_50, median_iou}
    """
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    device = next(model.parameters()).device

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            if tta:
                pred = tta_predict(model, batch_x, use_hflip=tta_hflip)
                all_preds.append(pred)
            else:
                outputs = model(batch_x)
                all_preds.append(outputs.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

    predictions = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    ious = compute_iou(predictions, targets)

    metrics = {
        'mean_iou': float(np.mean(ious)),
        'median_iou': float(np.median(ious)),
        'iou_at_50': float(np.mean(ious >= 0.5)),
        'iou_at_30': float(np.mean(ious >= 0.3)),
        'n_samples': len(ious),
    }

    return metrics, predictions, targets


def generate_bbox_report(metrics, output_dir, predictions=None, targets=None):
    """Generate evaluation report with IoU histogram and summary JSON.

    Args:
        metrics: dict from evaluate_bbox_model
        output_dir: Directory to save output files
        predictions: (N, 4) predicted boxes
        targets: (N, 4) target boxes
    """
    os.makedirs(output_dir, exist_ok=True)

    # IoU histogram
    if predictions is not None and targets is not None:
        ious = compute_iou(predictions, targets)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(ious, bins=20, range=(0, 1), edgecolor='black', alpha=0.7)
        ax.axvline(x=0.5, color='r', linestyle='--', label='IoU=0.5 threshold')
        ax.axvline(x=0.3, color='orange', linestyle='--', label='IoU=0.3 threshold')
        ax.set_xlabel('IoU')
        ax.set_ylabel('Count')
        ax.set_title(f'IoU Distribution (mean={metrics["mean_iou"]:.3f}, '
                     f'IoU@0.5={metrics["iou_at_50"]:.1%})')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'iou_histogram.png'))
        plt.close()

        # Scatter: predicted vs actual bbox center
        pred_cx = (predictions[:, 0] + predictions[:, 2]) / 2
        pred_cy = (predictions[:, 1] + predictions[:, 3]) / 2
        true_cx = (targets[:, 0] + targets[:, 2]) / 2
        true_cy = (targets[:, 1] + targets[:, 3]) / 2

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].scatter(true_cx, pred_cx, alpha=0.5, s=20)
        axes[0].plot([0, 1], [0, 1], 'r--')
        axes[0].set_xlabel('Actual center X')
        axes[0].set_ylabel('Predicted center X')
        axes[0].set_title('Center X: Predicted vs Actual')

        axes[1].scatter(true_cy, pred_cy, alpha=0.5, s=20)
        axes[1].plot([0, 1], [0, 1], 'r--')
        axes[1].set_xlabel('Actual center Y')
        axes[1].set_ylabel('Predicted center Y')
        axes[1].set_title('Center Y: Predicted vs Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'center_scatter.png'))
        plt.close()

    # Summary JSON
    with open(os.path.join(output_dir, 'evaluation_summary.json'), 'w') as f:
        json.dump(metrics, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Storm Bbox Model')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--tta', action='store_true',
                        help='Enable test-time augmentation (4 rotations)')
    parser.add_argument('--tta_hflip', action='store_true',
                        help='With --tta, also include hflip variants (8-way)')

    args = parser.parse_args()

    from v2.models import StormBboxNet
    from v2.dataset import create_bbox_dataloaders

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StormBboxNet(pretrained=False).to(device)
    _, _, test_loader = create_bbox_dataloaders(
        args.data_dir, batch_size=16, augment=False)

    metrics, predictions, targets = evaluate_bbox_model(
        model, test_loader, args.checkpoint,
        tta=args.tta, tta_hflip=args.tta_hflip)

    print("Evaluation Metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    generate_bbox_report(metrics, args.output_dir, predictions, targets)
    print(f"Report saved to {args.output_dir}")


if __name__ == '__main__':
    main()
