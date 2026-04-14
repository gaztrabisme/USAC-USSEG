"""
Typhoon Intensity Estimation ML Pipeline - Evaluation Module
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def dvorak_lookup(t_numbers):
    """Map T-number values to (wind_knots, pressure_hpa) via Dvorak table.

    Args:
        t_numbers: numpy array of T-number values

    Returns:
        (wind_knots, pressure_hpa) as numpy arrays
    """
    t = np.asarray(t_numbers, dtype=np.float64)
    wind = np.empty_like(t)
    pressure = np.empty_like(t)

    bins = [
        (1.5, 25, 1009),
        (2.5, 30, 1000),
        (3.5, 45, 991),
        (4.5, 65, 976),
        (5.5, 90, 954),
        (6.5, 115, 927),
        (7.5, 140, 898),
    ]

    wind[:] = 170
    pressure[:] = 858
    for threshold, w, p in reversed(bins):
        mask = t < threshold
        wind[mask] = w
        pressure[mask] = p

    return wind, pressure


def fit_calibrator(model, val_loader):
    """Fit isotonic regression calibrator on validation predictions.

    Learns a monotonic mapping from raw predicted T-numbers to calibrated
    T-numbers that corrects systematic bias (e.g., over-prediction at low
    intensity, under-prediction at high intensity).

    Args:
        model: Trained PyTorch model (already on device, eval mode)
        val_loader: Validation data loader

    Returns:
        sklearn IsotonicRegression fitted on val predictions → val targets
    """
    device = next(model.parameters()).device
    all_preds, all_targets = [], []

    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

    pred_t = np.concatenate(all_preds)[:, 0]
    true_t = np.concatenate(all_targets)[:, 0]

    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(pred_t, true_t)
    return calibrator


def evaluate_model(model, test_loader, checkpoint_path, calibrator=None):
    """Evaluate the model and compute per-target metrics.

    Handles both 3-output models (t_number, wind, pressure) and 1-output
    models (t_number only, with wind/pressure derived via Dvorak lookup).

    Args:
        model: PyTorch model (will be replaced with checkpoint weights)
        test_loader: Test data loader
        checkpoint_path: Path to checkpoint file (key: model_state_dict)
        calibrator: Optional IsotonicRegression for T-number calibration

    Returns:
        (metrics, predictions, targets) where metrics is
        {t_number: {mae, rmse, r2}, wind: {mae, rmse, r2}, pressure: {mae, rmse, r2}}
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
            batch_y = batch_y.to(device)
            outputs = model(batch_x)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

    predictions = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Detect single-output model
    single_output = predictions.ndim == 2 and predictions.shape[1] == 1

    if single_output:
        pred_t = predictions[:, 0]
        true_t = targets[:, 0] if targets.ndim == 2 else targets.ravel()

        # Apply isotonic calibration if provided
        if calibrator is not None:
            pred_t = calibrator.predict(pred_t)

        pred_wind, pred_pressure = dvorak_lookup(pred_t)
        true_wind, true_pressure = dvorak_lookup(true_t)

        # Build (N, 3) arrays for scatter plots
        predictions = np.column_stack([pred_t, pred_wind, pred_pressure])
        targets = np.column_stack([true_t, true_wind, true_pressure])

    metrics = {}
    target_names = ['t_number', 'wind', 'pressure']

    for i, target_name in enumerate(target_names):
        y_true = targets[:, i]
        y_pred = predictions[:, i]

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        metrics[target_name] = {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2)
        }

    return metrics, predictions, targets


def generate_report(metrics, output_dir, predictions=None, targets=None):
    """Generate evaluation report with scatter plots and JSON summary.

    Args:
        metrics: dict with per-target metrics {t_number: {mae, rmse, r2}, ...}
        output_dir: Directory to save output files
        predictions: optional numpy array (N, 3) of model predictions
        targets: optional numpy array (N, 3) of ground truth values
    """
    os.makedirs(output_dir, exist_ok=True)

    target_names = ['t_number', 'wind', 'pressure']
    target_labels = ['T-number', 'Wind Speed (knots)', 'Pressure (hPa)']

    for i, (target, label) in enumerate(zip(target_names, target_labels)):
        _fig, ax = plt.subplots(figsize=(8, 8))
        target_metrics = metrics.get(target, {})

        if predictions is not None and targets is not None:
            # Real scatter plot: predicted vs actual
            y_true = targets[:, i]
            y_pred = predictions[:, i]
            ax.scatter(y_true, y_pred, alpha=0.5, s=20)
            lo = min(y_true.min(), y_pred.min())
            hi = max(y_true.max(), y_pred.max())
            ax.plot([lo, hi], [lo, hi], 'r--', label='Perfect prediction')
            ax.set_xlabel(f'Actual {label}')
            ax.set_ylabel(f'Predicted {label}')
            mae = target_metrics.get('mae', 0)
            r2 = target_metrics.get('r2', 0)
            ax.set_title(f'{label}: MAE={mae:.3f}, R²={r2:.3f}')
            ax.legend()
        else:
            # Metric summary bar chart when no raw predictions available
            metric_names = ['mae', 'rmse', 'r2']
            values = [target_metrics.get(m, 0) for m in metric_names]
            x = np.arange(len(metric_names))
            ax.bar(x, values, color=['#3498db', '#e74c3c', '#2ecc71'])
            ax.set_xticks(x)
            ax.set_xticklabels(metric_names)
            ax.set_title(f'{target} Metrics')
            ax.set_ylabel('Value')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{target}_scatter.png'))
        plt.close()

    # Save evaluation summary JSON
    summary_path = os.path.join(output_dir, 'evaluation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(metrics, f, indent=2)


def main():
    """CLI entry point for evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate Typhoon Intensity Estimation Model')
    parser.add_argument('--model', type=str, required=True, choices=['mlp', 'cnn', 'cnnv2'],
                        help='Model type: mlp, cnn, or cnnv2')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to test data directory')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save evaluation results')
    parser.add_argument('--calibrate', action='store_true',
                        help='Apply isotonic calibration (fit on val set, apply to test)')

    args = parser.parse_args()

    from models import TyphoonMLP, TyphoonCNN, TyphoonCNNv2
    from dataset import create_dataloaders

    if args.model == 'mlp':
        model = TyphoonMLP()
    elif args.model == 'cnn':
        model = TyphoonCNN()
    else:
        model = TyphoonCNNv2()

    _, val_loader, test_loader = create_dataloaders(args.data_dir)

    # Fit calibrator on validation set if requested
    calibrator = None
    if args.calibrate:
        checkpoint = torch.load(args.checkpoint, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        calibrator = fit_calibrator(model, val_loader)
        print("Fitted isotonic calibrator on validation set")

    metrics, predictions, targets = evaluate_model(
        model, test_loader, args.checkpoint, calibrator=calibrator)
    print("Evaluation Metrics:")
    for target, target_metrics in metrics.items():
        print(f"  {target}: MAE={target_metrics['mae']:.4f}, RMSE={target_metrics['rmse']:.4f}, R2={target_metrics['r2']:.4f}")

    generate_report(metrics, args.output_dir, predictions=predictions, targets=targets)
    print(f"Report saved to {args.output_dir}")


if __name__ == '__main__':
    main()
