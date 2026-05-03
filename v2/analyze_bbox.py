"""
Stratified error analysis for storm bounding box model.

Decomposes IoU into position error vs size error, buckets by
storm size and disk position, generates diagnostic plots.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from v2.dataset import create_bbox_dataloaders
from v2.evaluate import compute_iou
from v2.models import StormBboxNet


def collect_predictions(model, test_loader, checkpoint_path):
    device = next(model.parameters()).device
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

    return np.concatenate(all_preds), np.concatenate(all_targets)


def decompose_errors(preds, targets):
    """Decompose bbox error into position (center) and size (w/h) components."""
    pred_cx = (preds[:, 0] + preds[:, 2]) / 2
    pred_cy = (preds[:, 1] + preds[:, 3]) / 2
    pred_w = preds[:, 2] - preds[:, 0]
    pred_h = preds[:, 3] - preds[:, 1]

    true_cx = (targets[:, 0] + targets[:, 2]) / 2
    true_cy = (targets[:, 1] + targets[:, 3]) / 2
    true_w = targets[:, 2] - targets[:, 0]
    true_h = targets[:, 3] - targets[:, 1]

    center_dist = np.sqrt((pred_cx - true_cx)**2 + (pred_cy - true_cy)**2)
    size_err_w = np.abs(pred_w - true_w)
    size_err_h = np.abs(pred_h - true_h)
    size_err = np.sqrt(size_err_w**2 + size_err_h**2)

    return {
        'center_dist': center_dist,
        'size_err': size_err,
        'pred_cx': pred_cx, 'pred_cy': pred_cy,
        'pred_w': pred_w, 'pred_h': pred_h,
        'true_cx': true_cx, 'true_cy': true_cy,
        'true_w': true_w, 'true_h': true_h,
    }


def bucket_analysis(ious, values, bucket_edges, label):
    """Compute mean IoU per bucket defined by value edges."""
    results = []
    for i in range(len(bucket_edges) - 1):
        lo, hi = bucket_edges[i], bucket_edges[i + 1]
        mask = (values >= lo) & (values < hi)
        n = mask.sum()
        if n > 0:
            results.append({
                'range': f'{lo:.3f}-{hi:.3f}',
                'n': int(n),
                'mean_iou': float(np.mean(ious[mask])),
                'median_iou': float(np.median(ious[mask])),
                'iou_at_50': float(np.mean(ious[mask] >= 0.5)),
            })
    return results


def run_analysis(preds, targets, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    ious = compute_iou(preds, targets)
    decomp = decompose_errors(preds, targets)

    # --- Storm size buckets (by target bbox area) ---
    true_area = decomp['true_w'] * decomp['true_h']
    area_edges = [0, 0.003, 0.006, 0.012, 0.05, 1.0]
    area_labels = ['tiny (<0.3%)', 'small (0.3-0.6%)', 'medium (0.6-1.2%)',
                   'large (1.2-5%)', 'xlarge (>5%)']
    size_buckets = bucket_analysis(ious, true_area, area_edges, 'area')

    # --- Disk position buckets (by distance from center) ---
    dist_from_center = np.sqrt((decomp['true_cx'] - 0.5)**2 +
                               (decomp['true_cy'] - 0.5)**2)
    pos_edges = [0, 0.1, 0.2, 0.3, 0.5, 1.0]
    pos_labels = ['center', 'near-center', 'mid', 'edge', 'far-edge']
    pos_buckets = bucket_analysis(ious, dist_from_center, pos_edges, 'position')

    # --- Position vs size error contribution ---
    corr_center = float(np.corrcoef(decomp['center_dist'], ious)[0, 1])
    corr_size = float(np.corrcoef(decomp['size_err'], ious)[0, 1])

    summary = {
        'n_samples': len(ious),
        'mean_iou': float(np.mean(ious)),
        'median_iou': float(np.median(ious)),
        'iou_at_50': float(np.mean(ious >= 0.5)),
        'iou_at_30': float(np.mean(ious >= 0.3)),
        'mean_center_dist': float(np.mean(decomp['center_dist'])),
        'mean_size_err': float(np.mean(decomp['size_err'])),
        'corr_center_dist_vs_iou': corr_center,
        'corr_size_err_vs_iou': corr_size,
        'size_buckets': size_buckets,
        'position_buckets': pos_buckets,
    }

    with open(os.path.join(output_dir, 'stratified_analysis.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # --- Plots ---

    # 1. IoU by storm size bucket
    fig, ax = plt.subplots(figsize=(10, 6))
    names = [f"{area_labels[i]}\nn={b['n']}" for i, b in enumerate(size_buckets)]
    means = [b['mean_iou'] for b in size_buckets]
    ax.bar(names, means, color='steelblue', edgecolor='black')
    ax.axhline(y=0.3, color='orange', linestyle='--', label='IoU=0.3')
    ax.axhline(y=0.5, color='red', linestyle='--', label='IoU=0.5')
    ax.set_ylabel('Mean IoU')
    ax.set_title('Mean IoU by Storm Size Bucket')
    ax.legend()
    ax.set_ylim(0, 0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'iou_by_size.png'), dpi=150)
    plt.close()

    # 2. IoU by disk position bucket
    fig, ax = plt.subplots(figsize=(10, 6))
    names = [f"{pos_labels[i]}\nn={b['n']}" for i, b in enumerate(pos_buckets)]
    means = [b['mean_iou'] for b in pos_buckets]
    ax.bar(names, means, color='coral', edgecolor='black')
    ax.axhline(y=0.3, color='orange', linestyle='--', label='IoU=0.3')
    ax.axhline(y=0.5, color='red', linestyle='--', label='IoU=0.5')
    ax.set_ylabel('Mean IoU')
    ax.set_title('Mean IoU by Disk Position (distance from center)')
    ax.legend()
    ax.set_ylim(0, 0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'iou_by_position.png'), dpi=150)
    plt.close()

    # 3. Center error vs size error scatter, colored by IoU
    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(decomp['center_dist'], decomp['size_err'],
                    c=ious, cmap='RdYlGn', s=20, alpha=0.7, vmin=0, vmax=1)
    plt.colorbar(sc, label='IoU')
    ax.set_xlabel('Center distance error (normalized)')
    ax.set_ylabel('Size error (normalized)')
    ax.set_title(f'Position vs Size Error\n'
                 f'corr(center,IoU)={corr_center:.2f}, '
                 f'corr(size,IoU)={corr_size:.2f}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'position_vs_size_error.png'), dpi=150)
    plt.close()

    # 4. Predicted vs actual bbox size scatter
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].scatter(decomp['true_w'], decomp['pred_w'], alpha=0.5, s=20)
    axes[0].plot([0, 0.4], [0, 0.4], 'r--')
    axes[0].set_xlabel('Actual width')
    axes[0].set_ylabel('Predicted width')
    axes[0].set_title('Width: Predicted vs Actual')

    axes[1].scatter(decomp['true_h'], decomp['pred_h'], alpha=0.5, s=20)
    axes[1].plot([0, 0.4], [0, 0.4], 'r--')
    axes[1].set_xlabel('Actual height')
    axes[1].set_ylabel('Predicted height')
    axes[1].set_title('Height: Predicted vs Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'size_scatter.png'), dpi=150)
    plt.close()

    return summary


def main():
    parser = argparse.ArgumentParser(description='Stratified Bbox Error Analysis')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StormBboxNet(pretrained=False).to(device)
    _, _, test_loader = create_bbox_dataloaders(
        args.data_dir, batch_size=16, augment=False)

    preds, targets = collect_predictions(model, test_loader, args.checkpoint)
    summary = run_analysis(preds, targets, args.output_dir)

    print("\n=== Stratified Error Analysis ===")
    print(f"Samples: {summary['n_samples']}")
    print(f"Mean IoU: {summary['mean_iou']:.3f}")
    print(f"Mean center error: {summary['mean_center_dist']:.4f}")
    print(f"Mean size error: {summary['mean_size_err']:.4f}")
    print(f"Corr(center_dist, IoU): {summary['corr_center_dist_vs_iou']:.3f}")
    print(f"Corr(size_err, IoU): {summary['corr_size_err_vs_iou']:.3f}")

    print("\n--- By Storm Size ---")
    for b in summary['size_buckets']:
        print(f"  {b['range']:>15s}  n={b['n']:3d}  "
              f"IoU={b['mean_iou']:.3f}  IoU@0.5={b['iou_at_50']:.1%}")

    print("\n--- By Disk Position ---")
    for b in summary['position_buckets']:
        print(f"  {b['range']:>15s}  n={b['n']:3d}  "
              f"IoU={b['mean_iou']:.3f}  IoU@0.5={b['iou_at_50']:.1%}")


if __name__ == '__main__':
    main()
