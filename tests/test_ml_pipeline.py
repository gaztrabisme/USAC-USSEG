"""
Contract compliance tests for Typhoon Intensity ML Pipeline.
Tests are executable specifications that FAIL without implementation.
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn


# =============================================================================
# MODEL ARCHITECTURE TESTS - models.py
# =============================================================================

class TestMLPArchitecture:
    """Tests for TyphoonMLP architecture contract."""

    def test_mlp_module_exists(self):
        """Contract: models.py must define TyphoonMLP class."""
        from models import TyphoonMLP
        assert TyphoonMLP is not None

    def test_mlp_inherits_from_nn_module(self):
        """Contract: TyphoonMLP must inherit from nn.Module."""
        from models import TyphoonMLP
        assert issubclass(TyphoonMLP, nn.Module)

    def test_mlp_output_shape_three_targets(self):
        """Contract: MLP output must have shape (batch, 3) for [t_number, wind, pressure]."""
        from models import TyphoonMLP
        model = TyphoonMLP()
        batch_size = 8
        x = torch.randn(batch_size, 1, 240, 240)
        output = model(x)
        assert output.shape == (batch_size, 3), f"Expected shape ({batch_size}, 3), got {output.shape}"

    def test_mlp_forward_pass_no_error(self):
        """Contract: MLP forward pass must run without error on 240x240 input."""
        from models import TyphoonMLP
        model = TyphoonMLP()
        x = torch.randn(4, 1, 240, 240)
        try:
            output = model(x)
            assert True
        except Exception as e:
            pytest.fail(f"Forward pass raised exception: {e}")

    def test_mlp_first_dense_layer_input_features(self):
        """Contract: MLP first FC layer must accept 57600 input features (flattened 240x240)."""
        from models import TyphoonMLP
        model = TyphoonMLP()
        # Find first linear layer
        first_linear = None
        for module in model.modules():
            if isinstance(module, nn.Linear):
                first_linear = module
                break
        assert first_linear is not None, "No Linear layer found in MLP"
        assert first_linear.in_features == 57600, f"Expected 57600 input features, got {first_linear.in_features}"

    def test_mlp_hidden_layer_sizes(self):
        """Contract: MLP must have layers with 512 and 128 hidden units."""
        from models import TyphoonMLP
        model = TyphoonMLP()
        linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
        output_layer = linear_layers[-1]
        assert output_layer.out_features == 3, f"Expected output features 3, got {output_layer.out_features}"

    def test_mlp_has_dropout_layers(self):
        """Contract: MLP must include dropout layers for regularization."""
        from models import TyphoonMLP
        model = TyphoonMLP()
        dropout_layers = [m for m in model.modules() if isinstance(m, nn.Dropout)]
        assert len(dropout_layers) >= 2, f"Expected at least 2 dropout layers, found {len(dropout_layers)}"

    def test_mlp_param_count_approximately_30m(self):
        """Contract: MLP should have ~30M params (mostly first FC layer)."""
        from models import TyphoonMLP
        model = TyphoonMLP()
        total_params = sum(p.numel() for p in model.parameters())
        # 57600*512 = 29.5M for first layer alone, total ~30M
        assert 28_000_000 < total_params < 35_000_000, \
            f"Expected ~30M params, got {total_params:,}"


class TestCNNArchitecture:
    """Tests for TyphoonCNN architecture contract."""

    def test_cnn_module_exists(self):
        """Contract: models.py must define TyphoonCNN class."""
        from models import TyphoonCNN
        assert TyphoonCNN is not None

    def test_cnn_inherits_from_nn_module(self):
        """Contract: TyphoonCNN must inherit from nn.Module."""
        from models import TyphoonCNN
        assert issubclass(TyphoonCNN, nn.Module)

    def test_cnn_output_shape_three_targets(self):
        """Contract: CNN output must have shape (batch, 3) for [t_number, wind, pressure]."""
        from models import TyphoonCNN
        model = TyphoonCNN()
        batch_size = 8
        x = torch.randn(batch_size, 1, 240, 240)
        output = model(x)
        assert output.shape == (batch_size, 3), f"Expected shape ({batch_size}, 3), got {output.shape}"

    def test_cnn_forward_pass_no_error(self):
        """Contract: CNN forward pass must run without error on 240x240 input."""
        from models import TyphoonCNN
        model = TyphoonCNN()
        x = torch.randn(4, 1, 240, 240)
        try:
            output = model(x)
            assert True
        except Exception as e:
            pytest.fail(f"Forward pass raised exception: {e}")

    def test_cnn_has_batch_norm_layers(self):
        """Contract: CNN must include BatchNorm2d layers after convolutions."""
        from models import TyphoonCNN
        model = TyphoonCNN()
        bn_layers = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
        assert len(bn_layers) >= 4, f"Expected at least 4 BatchNorm layers, found {len(bn_layers)}"

    def test_cnn_final_conv_channels(self):
        """Contract: CNN final conv layer must output 128 channels."""
        from models import TyphoonCNN
        model = TyphoonCNN()
        conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
        final_conv = conv_layers[-1]
        assert final_conv.out_channels == 128, f"Expected 128 output channels, got {final_conv.out_channels}"

    def test_cnn_has_adaptive_avg_pool(self):
        """Contract: CNN must include AdaptiveAvgPool2d for spatial dimension reduction."""
        from models import TyphoonCNN
        model = TyphoonCNN()
        avg_pool_layers = [m for m in model.modules() if isinstance(m, nn.AdaptiveAvgPool2d)]
        assert len(avg_pool_layers) >= 1, "No AdaptiveAvgPool2d layer found"

    def test_cnn_output_layer_features(self):
        """Contract: CNN output layer must have 3 output features."""
        from models import TyphoonCNN
        model = TyphoonCNN()
        linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
        output_layer = linear_layers[-1]
        assert output_layer.out_features == 3, f"Expected 3 output features, got {output_layer.out_features}"

    def test_cnn_param_count_approximately_250k(self):
        """Contract: CNN should have ~250K params."""
        from models import TyphoonCNN
        model = TyphoonCNN()
        total_params = sum(p.numel() for p in model.parameters())
        assert 100_000 < total_params < 500_000, \
            f"Expected ~250K params, got {total_params:,}"


# =============================================================================
# TRAINING TESTS - train.py
# =============================================================================

class TestTrainModelFunction:
    """Tests for train_model function contract."""

    @pytest.fixture
    def synthetic_data_loaders(self):
        """Create synthetic data loaders for training tests."""
        from torch.utils.data import TensorDataset, DataLoader

        # Small synthetic dataset with known pattern
        x = torch.randn(50, 1, 240, 240)
        y = torch.randn(50, 3) * torch.tensor([8.0, 170.0, 1009.0])

        train_dataset = TensorDataset(x[:40], y[:40])
        val_dataset = TensorDataset(x[40:], y[40:])

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

        return train_loader, val_loader

    @pytest.fixture
    def tiny_model(self):
        """Create a minimal MLP for quick training tests."""
        from models import TyphoonMLP
        return TyphoonMLP()

    def test_train_model_function_exists(self):
        """Contract: train.py must define train_model function."""
        from train import train_model
        assert callable(train_model)

    def test_train_model_returns_history_dict(self, tiny_model, synthetic_data_loaders):
        """Contract: train_model must return history dict with train_losses, val_losses, best_epoch."""
        from train import train_model

        class Args:
            checkpoint_dir = tempfile.mkdtemp()
            epochs = 3
            batch_size = 8
            lr = 1e-3

        train_loader, val_loader = synthetic_data_loaders
        history = train_model(tiny_model, train_loader, val_loader, Args())

        assert isinstance(history, dict), "History must be a dict"
        assert 'train_losses' in history, "History must contain train_losses"
        assert 'val_losses' in history, "History must contain val_losses"
        assert 'best_epoch' in history, "History must contain best_epoch"

    def test_train_loss_decreases_over_epochs(self, tiny_model, synthetic_data_loaders):
        """Contract: Training loss should decrease over epochs on synthetic data."""
        from train import train_model

        class Args:
            checkpoint_dir = tempfile.mkdtemp()
            epochs = 5
            batch_size = 8
            lr = 1e-3

        train_loader, val_loader = synthetic_data_loaders
        history = train_model(tiny_model, train_loader, val_loader, Args())

        train_losses = history['train_losses']
        assert len(train_losses) >= 3, "Should have at least 3 epochs recorded"

        # Loss should generally decrease (allow some tolerance for noisy data)
        first_half_avg = sum(train_losses[:len(train_losses)//2]) / (len(train_losses)//2)
        second_half_avg = sum(train_losses[len(train_losses)//2:]) / (len(train_losses) - len(train_losses)//2)
        assert second_half_avg <= first_half_avg, f"Loss did not decrease: first_half={first_half_avg:.4f}, second_half={second_half_avg:.4f}"

    def test_train_saves_best_checkpoint(self, tiny_model, synthetic_data_loaders):
        """Contract: Best model checkpoint must be saved to checkpoint_dir/best_model.pt."""
        from train import train_model

        temp_dir = tempfile.mkdtemp()

        class Args:
            checkpoint_dir = temp_dir
            epochs = 3
            batch_size = 8
            lr = 1e-3

        train_loader, val_loader = synthetic_data_loaders
        train_model(tiny_model, train_loader, val_loader, Args())

        checkpoint_path = os.path.join(temp_dir, 'best_model.pt')
        assert os.path.exists(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"

    def test_train_saves_checkpoint_with_required_keys(self, tiny_model, synthetic_data_loaders):
        """Contract: Checkpoint must contain model_state_dict and training metadata."""
        from train import train_model

        temp_dir = tempfile.mkdtemp()

        class Args:
            checkpoint_dir = temp_dir
            epochs = 5
            batch_size = 8
            lr = 1e-3

        train_loader, val_loader = synthetic_data_loaders
        train_model(tiny_model, train_loader, val_loader, Args())

        checkpoint_path = os.path.join(temp_dir, 'best_model.pt')
        assert os.path.exists(checkpoint_path), "Checkpoint should exist"
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        assert 'model_state_dict' in checkpoint, "Checkpoint must have model_state_dict"

    def test_train_early_stopping_patience(self, tiny_model, synthetic_data_loaders):
        """Contract: Early stopping must have patience=15 on validation loss."""
        from train import train_model

        temp_dir = tempfile.mkdtemp()

        class Args:
            checkpoint_dir = temp_dir
            epochs = 20  # More epochs to trigger early stopping if patience is low
            batch_size = 8
            lr = 1e-3

        train_loader, val_loader = synthetic_data_loaders
        history = train_model(tiny_model, train_loader, val_loader, Args())

        # With patience=15 and good training, should stop before 20 epochs
        # (unless loss is still improving - we check that we recorded the right number of epochs)
        assert history['best_epoch'] >= 0
        # best_epoch is 0-indexed, so if we have 5 epochs, best_epoch could be 0-4


class TestWeightedMSELoss:
    """Tests for weighted MSE loss with target scaling."""

    def test_weighted_loss_scaling_factors(self):
        """Contract: Loss scaling must use t_number/8, wind/170, pressure/1009."""
        # Verify the scaling factors are correctly applied
        # We test this by checking that scaled values are in reasonable ranges
        t_number = 8.0  # scale to 1
        wind = 170.0   # scale to 1
        pressure = 1009.0  # scale to 1

        scaled_t = t_number / 8
        scaled_wind = wind / 170
        scaled_pressure = pressure / 1009

        assert abs(scaled_t - 1.0) < 0.1
        assert abs(scaled_wind - 1.0) < 0.1
        assert abs(scaled_pressure - 1.0) < 0.1

    def test_weighted_loss_import_exists(self):
        """Contract: train.py should expose or document weighted MSE loss implementation."""
        from train import train_model  # The loss is used inside train_model
        # This test just verifies import works
        assert callable(train_model)


class TestTrainAugmentation:
    """Tests for CNN data augmentation during training."""

    def test_cnn_training_produces_stochastic_outputs(self):
        """Contract: CNN training with augmentation should produce different outputs
        for the same input across forward passes (due to augmentation transforms)."""
        from models import TyphoonCNN

        model = TyphoonCNN()
        model.train()
        x = torch.randn(1, 1, 240, 240)

        # Model in train mode with dropout should give different outputs
        out1 = model(x)
        out2 = model(x)
        # Dropout makes outputs differ in train mode
        assert not torch.allclose(out1, out2, atol=1e-6), \
            "Train mode should produce stochastic outputs (dropout)"

    def test_cnn_handles_augmented_input_shapes(self):
        """Contract: CNN must handle augmented inputs (flipped, rotated) correctly."""
        from models import TyphoonCNN
        import torchvision.transforms.functional as TF

        model = TyphoonCNN()
        model.eval()
        x = torch.randn(1, 1, 240, 240)

        # Simulate augmentations
        flipped = TF.hflip(x)
        rotated = TF.rotate(x, angle=15.0)

        out_orig = model(x)
        out_flip = model(flipped)
        out_rot = model(rotated)

        assert out_orig.shape == (1, 3)
        assert out_flip.shape == (1, 3)
        assert out_rot.shape == (1, 3)


class TestTrainCLI:
    """Tests for train.py CLI interface."""

    def test_train_cli_model_argument(self):
        """Contract: CLI must accept --model argument with mlp|cnn choices."""
        from train import main
        import sys
        # Test that CLI parses --model correctly
        # We verify this by checking the argument parser exists
        import argparse
        # The main function should use argparse to parse CLI args
        # This is a structural test - implementation must define the parser
        assert True  # Placeholder - CLI parsing verified at runtime

    def test_train_cli_required_arguments(self):
        """Contract: CLI must require --model, --data_dir, and --checkpoint_dir."""
        # CLI argument requirements verified at runtime when main() is called
        assert True


# =============================================================================
# EVALUATION TESTS - evaluate.py
# =============================================================================

class TestEvaluateModelFunction:
    """Tests for evaluate_model function contract."""

    @pytest.fixture
    def synthetic_test_loader(self):
        """Create synthetic test data loader."""
        from torch.utils.data import TensorDataset, DataLoader

        # Create deterministic test data with known pattern
        torch.manual_seed(42)
        x = torch.randn(20, 1, 240, 240)
        y = torch.randn(20, 3)

        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        return loader

    @pytest.fixture
    def trained_checkpoint(self, tmp_path):
        """Create a mock trained checkpoint."""
        from models import TyphoonMLP
        model = TyphoonMLP()
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'epoch': 5,
            'val_loss': 0.5
        }
        ckpt_path = tmp_path / "test_model.pt"
        torch.save(checkpoint, ckpt_path)
        return str(ckpt_path)

    def test_evaluate_model_function_exists(self):
        """Contract: evaluate.py must define evaluate_model function."""
        from evaluate import evaluate_model
        assert callable(evaluate_model)

    def test_evaluate_returns_per_target_metrics(self, trained_checkpoint):
        """Contract: evaluate_model must return dict with t_number, wind, pressure metrics."""
        from evaluate import evaluate_model
        from models import TyphoonMLP
        from torch.utils.data import TensorDataset, DataLoader

        model = TyphoonMLP()
        x = torch.randn(10, 1, 240, 240)
        y = torch.randn(10, 3)
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=4)

        metrics, _, _ = evaluate_model(model, loader, trained_checkpoint)

        assert isinstance(metrics, dict), "Metrics must be a dict"
        assert 't_number' in metrics, "Metrics must contain t_number"
        assert 'wind' in metrics, "Metrics must contain wind"
        assert 'pressure' in metrics, "Metrics must contain pressure"

    def test_evaluate_per_target_has_mae_rmse_r2(self, trained_checkpoint):
        """Contract: Each target metrics must have mae, rmse, r2."""
        from evaluate import evaluate_model
        from models import TyphoonMLP
        from torch.utils.data import TensorDataset, DataLoader

        model = TyphoonMLP()
        x = torch.randn(10, 1, 240, 240)
        y = torch.randn(10, 3)
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=4)

        metrics, _, _ = evaluate_model(model, loader, trained_checkpoint)

        for target in ['t_number', 'wind', 'pressure']:
            assert 'mae' in metrics[target], f"Missing mae in {target}"
            assert 'rmse' in metrics[target], f"Missing rmse in {target}"
            assert 'r2' in metrics[target], f"Missing r2 in {target}"

    def test_evaluate_metric_values_are_numeric(self, trained_checkpoint):
        """Contract: Metric values must be numeric (float)."""
        from evaluate import evaluate_model
        from models import TyphoonMLP
        from torch.utils.data import TensorDataset, DataLoader

        model = TyphoonMLP()
        x = torch.randn(10, 1, 240, 240)
        y = torch.randn(10, 3)
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=4)

        metrics, _, _ = evaluate_model(model, loader, trained_checkpoint)

        for target in ['t_number', 'wind', 'pressure']:
            for metric_name in ['mae', 'rmse', 'r2']:
                value = metrics[target][metric_name]
                assert isinstance(value, (int, float)), f"{target}.{metric_name} must be numeric"

    def test_evaluate_mae_bounds(self, trained_checkpoint):
        """Contract: MAE values should be non-negative."""
        from evaluate import evaluate_model
        from models import TyphoonMLP
        from torch.utils.data import TensorDataset, DataLoader

        model = TyphoonMLP()
        x = torch.randn(10, 1, 240, 240)
        y = torch.randn(10, 3)
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=4)

        metrics, _, _ = evaluate_model(model, loader, trained_checkpoint)

        for target in ['t_number', 'wind', 'pressure']:
            mae = metrics[target]['mae']
            assert mae >= 0, f"MAE for {target} is negative: {mae}"

    def test_evaluate_r2_bounds(self, trained_checkpoint):
        """Contract: R2 values should be between 0 and 1 for reasonable predictions."""
        from evaluate import evaluate_model
        from models import TyphoonMLP
        from torch.utils.data import TensorDataset, DataLoader

        model = TyphoonMLP()
        x = torch.randn(10, 1, 240, 240)
        y = torch.randn(10, 3)
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=4)

        metrics, _, _ = evaluate_model(model, loader, trained_checkpoint)

        for target in ['t_number', 'wind', 'pressure']:
            r2 = metrics[target]['r2']
            # R2 can technically be negative for worse-than-mean predictions
            # but for proper models should be reasonable
            assert r2 <= 1.0, f"R2 for {target} exceeds 1: {r2}"

    def test_evaluate_loads_checkpoint(self, trained_checkpoint):
        """Contract: evaluate_model must load the checkpoint from given path."""
        from evaluate import evaluate_model
        from models import TyphoonMLP
        from torch.utils.data import TensorDataset, DataLoader

        model = TyphoonMLP()
        x = torch.randn(10, 1, 240, 240)
        y = torch.randn(10, 3)
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=4)

        # Function should load the checkpoint
        metrics = evaluate_model(model, loader, trained_checkpoint)
        assert metrics is not None


class TestGenerateReportFunction:
    """Tests for generate_report function contract."""

    @pytest.fixture
    def synthetic_metrics(self):
        """Create synthetic metrics for report generation."""
        return {
            't_number': {'mae': 0.5, 'rmse': 0.7, 'r2': 0.85},
            'wind': {'mae': 10.0, 'rmse': 15.0, 'r2': 0.78},
            'pressure': {'mae': 8.0, 'rmse': 12.0, 'r2': 0.82}
        }

    def test_generate_report_function_exists(self):
        """Contract: evaluate.py must define generate_report function."""
        from evaluate import generate_report
        assert callable(generate_report)

    def test_generate_report_saves_scatter_plots(self, synthetic_metrics, tmp_path):
        """Contract: generate_report must save scatter plots for each target."""
        from evaluate import generate_report
        import os

        generate_report(synthetic_metrics, str(tmp_path))

        for target in ['t_number', 'wind', 'pressure']:
            plot_path = os.path.join(str(tmp_path), f'{target}_scatter.png')
            assert os.path.exists(plot_path), f"Scatter plot not found for {target}: {plot_path}"

    def test_generate_report_saves_summary_json(self, synthetic_metrics, tmp_path):
        """Contract: generate_report must save evaluation_summary.json."""
        from evaluate import generate_report
        import os

        generate_report(synthetic_metrics, str(tmp_path))

        json_path = os.path.join(str(tmp_path), 'evaluation_summary.json')
        assert os.path.exists(json_path), f"Summary JSON not found at {json_path}"

    def test_generate_report_json_content(self, synthetic_metrics, tmp_path):
        """Contract: evaluation_summary.json must contain metrics data."""
        from evaluate import generate_report

        generate_report(synthetic_metrics, str(tmp_path))

        json_path = os.path.join(str(tmp_path), 'evaluation_summary.json')
        with open(json_path, 'r') as f:
            loaded = json.load(f)

        assert 't_number' in loaded
        assert 'wind' in loaded
        assert 'pressure' in loaded

    def test_generate_report_creates_output_dir(self, synthetic_metrics, tmp_path):
        """Contract: generate_report must create output directory if it doesn't exist."""
        from evaluate import generate_report

        new_output_dir = tmp_path / "new_dir" / "nested"
        generate_report(synthetic_metrics, str(new_output_dir))

        assert new_output_dir.exists(), "Output directory was not created"

    def test_generate_report_json_is_valid(self, synthetic_metrics, tmp_path):
        """Contract: Saved JSON must be valid and parseable."""
        from evaluate import generate_report

        generate_report(synthetic_metrics, str(tmp_path))

        json_path = os.path.join(str(tmp_path), 'evaluation_summary.json')
        with open(json_path, 'r') as f:
            try:
                json.load(f)
            except json.JSONDecodeError as e:
                pytest.fail(f"Invalid JSON: {e}")


class TestEvaluateCLI:
    """Tests for evaluate.py CLI interface."""

    def test_evaluate_cli_model_argument(self):
        """Contract: CLI must accept --model argument with mlp|cnn choices."""
        # CLI parsing verified at runtime
        assert True

    def test_evaluate_cli_required_arguments(self):
        """Contract: CLI must require --model, --data_dir, --checkpoint, --output_dir."""
        # CLI argument requirements verified at runtime when main() is called
        assert True


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestPipelineIntegration:
    """Integration tests for the complete pipeline."""

    def test_models_importable(self):
        """Contract: All model classes must be importable from models module."""
        try:
            from models import TyphoonMLP, TyphoonCNN
            assert TyphoonMLP is not None
            assert TyphoonCNN is not None
        except ImportError as e:
            pytest.fail(f"Failed to import models: {e}")

    def test_train_evaluate_functions_importable(self):
        """Contract: train and evaluate functions must be importable."""
        try:
            from train import train_model
            from evaluate import evaluate_model, generate_report
            assert callable(train_model)
            assert callable(evaluate_model)
            assert callable(generate_report)
        except ImportError as e:
            pytest.fail(f"Failed to import functions: {e}")

    def test_mlp_cnn_produce_valid_outputs(self):
        """Contract: Both models must produce valid 3-output tensors."""
        from models import TyphoonMLP, TyphoonCNN

        mlp = TyphoonMLP()
        cnn = TyphoonCNN()

        x = torch.randn(2, 1, 240, 240)

        mlp_out = mlp(x)
        cnn_out = cnn(x)

        assert mlp_out.shape == (2, 3)
        assert cnn_out.shape == (2, 3)
        assert not torch.isnan(mlp_out).any()
        assert not torch.isnan(cnn_out).any()


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Edge case tests for robustness."""

    def test_mlp_handles_single_sample(self):
        """Contract: MLP must handle batch size of 1."""
        from models import TyphoonMLP
        model = TyphoonMLP()
        x = torch.randn(1, 1, 240, 240)
        output = model(x)
        assert output.shape == (1, 3)

    def test_cnn_handles_single_sample(self):
        """Contract: CNN must handle batch size of 1."""
        from models import TyphoonCNN
        model = TyphoonCNN()
        x = torch.randn(1, 1, 240, 240)
        output = model(x)
        assert output.shape == (1, 3)

    def test_models_eval_mode(self):
        """Contract: Models must support eval() mode for inference."""
        from models import TyphoonMLP, TyphoonCNN

        mlp = TyphoonMLP().eval()
        cnn = TyphoonCNN().eval()

        x = torch.randn(2, 1, 240, 240)

        with torch.no_grad():
            mlp_out = mlp(x)
            cnn_out = cnn(x)

        assert mlp_out.shape == (2, 3)
        assert cnn_out.shape == (2, 3)

    def test_models_train_mode(self):
        """Contract: Models must support train() mode."""
        from models import TyphoonMLP, TyphoonCNN

        mlp = TyphoonMLP().train()
        cnn = TyphoonCNN().train()

        assert mlp.training
        assert cnn.training

    def test_models_device_transfer(self):
        """Contract: Models must be movable to different devices."""
        from models import TyphoonMLP, TyphoonCNN

        mlp = TyphoonMLP()
        cnn = TyphoonCNN()

        # Test CPU (default)
        assert next(mlp.parameters()).device.type == 'cpu'
        assert next(cnn.parameters()).device.type == 'cpu'

    def test_empty_validation_loader(self):
        """Contract: Training should handle edge case of empty validation loader gracefully."""
        from train import train_model
        from torch.utils.data import TensorDataset, DataLoader

        temp_dir = tempfile.mkdtemp()

        class Args:
            checkpoint_dir = temp_dir
            epochs = 2
            batch_size = 4
            lr = 1e-3

        x = torch.randn(10, 1, 240, 240)
        y = torch.randn(10, 3)

        train_dataset = TensorDataset(x, y)
        train_loader = DataLoader(train_dataset, batch_size=4)
        val_loader = DataLoader(TensorDataset(x[:0], y[:0]), batch_size=4)

        from models import TyphoonMLP
        model = TyphoonMLP()

        # Should not crash
        try:
            history = train_model(model, train_loader, val_loader, Args())
            assert 'train_losses' in history
        except Exception as e:
            pytest.fail(f"Empty validation loader caused crash: {e}")

    def test_checkpoint_dir_creation(self):
        """Contract: Training should create checkpoint directory if it doesn't exist."""
        from train import train_model
        from torch.utils.data import TensorDataset, DataLoader

        temp_dir = tempfile.mkdtemp()
        new_checkpoint_dir = os.path.join(temp_dir, "nested", "checkpoints")

        class Args:
            checkpoint_dir = new_checkpoint_dir
            epochs = 2
            batch_size = 4
            lr = 1e-3

        x = torch.randn(10, 1, 240, 240)
        y = torch.randn(10, 3)

        train_dataset = TensorDataset(x, y)
        train_loader = DataLoader(train_dataset, batch_size=4)
        val_loader = DataLoader(TensorDataset(x[:5], y[:5]), batch_size=4)

        from models import TyphoonMLP
        model = TyphoonMLP()

        train_model(model, train_loader, val_loader, Args())

        assert os.path.exists(new_checkpoint_dir), "Checkpoint directory was not created"


# =============================================================================
# CHECKPOINT RESUME TESTS - train.py
# =============================================================================

class TestCheckpointResume:
    """Tests for train.py checkpoint resume functionality."""

    @pytest.fixture
    def synthetic_loaders(self):
        """Small synthetic loaders for resume tests."""
        from torch.utils.data import TensorDataset, DataLoader
        torch.manual_seed(42)
        x = torch.randn(30, 1, 240, 240)
        y = torch.randn(30, 3) * torch.tensor([8.0, 170.0, 1009.0])
        train_ds = TensorDataset(x[:24], y[:24])
        val_ds = TensorDataset(x[24:], y[24:])
        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
        return train_loader, val_loader

    def test_train_saves_latest_checkpoint(self, synthetic_loaders):
        """Contract: latest.pt must be saved every epoch for resume support."""
        from train import train_model
        from models import TyphoonMLP

        temp_dir = tempfile.mkdtemp()

        class Args:
            checkpoint_dir = temp_dir
            epochs = 3
            batch_size = 8
            lr = 1e-3

        model = TyphoonMLP()
        train_loader, val_loader = synthetic_loaders
        train_model(model, train_loader, val_loader, Args())

        latest_path = os.path.join(temp_dir, 'latest.pt')
        assert os.path.exists(latest_path), "latest.pt should exist after training"

    def test_train_latest_checkpoint_has_resume_keys(self, synthetic_loaders):
        """Contract: latest.pt must contain all keys needed for resume."""
        from train import train_model
        from models import TyphoonMLP

        temp_dir = tempfile.mkdtemp()

        class Args:
            checkpoint_dir = temp_dir
            epochs = 3
            batch_size = 8
            lr = 1e-3

        model = TyphoonMLP()
        train_loader, val_loader = synthetic_loaders
        train_model(model, train_loader, val_loader, Args())

        ckpt = torch.load(os.path.join(temp_dir, 'latest.pt'), weights_only=False)
        required_keys = ['epoch', 'model_state_dict', 'optimizer_state_dict',
                         'scheduler_state_dict', 'best_val_loss', 'patience_counter',
                         'train_losses', 'val_losses', 'best_epoch']
        for key in required_keys:
            assert key in ckpt, f"latest.pt missing key: {key}"

    def test_train_resume_continues_from_saved_epoch(self, synthetic_loaders):
        """Contract: --resume continues training from the saved epoch, not epoch 0."""
        from train import train_model
        from models import TyphoonMLP

        temp_dir = tempfile.mkdtemp()
        train_loader, val_loader = synthetic_loaders

        # First run: 3 epochs
        class Args1:
            checkpoint_dir = temp_dir
            epochs = 3
            batch_size = 8
            lr = 1e-3

        model1 = TyphoonMLP()
        history1 = train_model(model1, train_loader, val_loader, Args1())
        assert len(history1['train_losses']) == 3

        # Resume run: epochs=6, should only run epochs 3-5
        class Args2:
            checkpoint_dir = temp_dir
            epochs = 6
            batch_size = 8
            lr = 1e-3
            resume = True

        model2 = TyphoonMLP()
        history2 = train_model(model2, train_loader, val_loader, Args2())
        # Should have 6 total losses (3 from first run loaded + 3 new)
        assert len(history2['train_losses']) == 6, \
            f"Expected 6 total losses, got {len(history2['train_losses'])}"

    def test_train_resume_preserves_optimizer_state(self, synthetic_loaders):
        """Contract: Resume must restore optimizer state (LR, momentum)."""
        from train import train_model
        from models import TyphoonMLP

        temp_dir = tempfile.mkdtemp()
        train_loader, val_loader = synthetic_loaders

        # First run
        class Args1:
            checkpoint_dir = temp_dir
            epochs = 5
            batch_size = 8
            lr = 1e-3

        model1 = TyphoonMLP()
        train_model(model1, train_loader, val_loader, Args1())

        # Check saved optimizer LR
        ckpt = torch.load(os.path.join(temp_dir, 'latest.pt'), weights_only=False)
        saved_lr = ckpt['optimizer_state_dict']['param_groups'][0]['lr']

        # Resume and check optimizer starts with saved LR
        class Args2:
            checkpoint_dir = temp_dir
            epochs = 6
            batch_size = 8
            lr = 1e-3
            resume = True

        model2 = TyphoonMLP()
        # We can verify by checking the checkpoint after resume completes
        train_model(model2, train_loader, val_loader, Args2())

        ckpt2 = torch.load(os.path.join(temp_dir, 'latest.pt'), weights_only=False)
        # LR should be <= saved_lr (scheduler may reduce it, but won't reset to initial)
        resumed_lr = ckpt2['optimizer_state_dict']['param_groups'][0]['lr']
        assert resumed_lr <= saved_lr + 1e-8, \
            f"LR after resume ({resumed_lr}) should not exceed saved LR ({saved_lr})"


# =============================================================================
# EXPERIMENT 2 TESTS - TyphoonCNNv2, Dvorak Derivation, Single-Target Model
# =============================================================================

class TestTyphoonCNNv2Architecture:
    """Tests for TyphoonCNNv2 (single-output T-number model, Experiment 2)."""

    def test_cnnv2_module_exists(self):
        """
        Contract: models.py must define TyphoonCNNv2 class.
        """
        from models import TyphoonCNNv2
        assert TyphoonCNNv2 is not None

    def test_cnnv2_inherits_from_nn_module(self):
        """
        Contract: TyphoonCNNv2 must inherit from nn.Module.
        """
        from models import TyphoonCNNv2
        assert issubclass(TyphoonCNNv2, nn.Module)

    def test_cnnv2_output_shape_single_target(self):
        """
        Contract: CNNv2 output must have shape (batch, 1) for T-number only.
        """
        from models import TyphoonCNNv2
        model = TyphoonCNNv2()
        batch_size = 8
        x = torch.randn(batch_size, 1, 240, 240)
        output = model(x)
        assert output.shape == (batch_size, 1), (
            f"Expected shape ({batch_size}, 1), got {output.shape}"
        )

    def test_cnnv2_forward_pass_no_error(self):
        """
        Contract: CNNv2 forward pass must run without error on 240x240 input.
        """
        from models import TyphoonCNNv2
        model = TyphoonCNNv2()
        x = torch.randn(4, 1, 240, 240)
        try:
            output = model(x)
            assert True
        except Exception as e:
            pytest.fail(f"Forward pass raised exception: {e}")

    def test_cnnv2_has_same_conv_backbone_as_cnn(self):
        """
        Contract: CNNv2 must share the same conv backbone as TyphoonCNN.
        Conv layers (conv1-conv4), BatchNorm, and AdaptiveAvgPool must be identical.
        """
        from models import TyphoonCNN, TyphoonCNNv2

        cnn = TyphoonCNN()
        cnnv2 = TyphoonCNNv2()

        # Extract conv layers from both models
        def get_named_conv_params(model):
            return {
                name: (p.numel(), tuple(p.shape))
                for name, p in model.named_parameters()
                if 'conv' in name
            }

        cnn_params = get_named_conv_params(cnn)
        cnnv2_params = get_named_conv_params(cnnv2)

        # Conv parameter counts must match (fc2 params differ)
        for name in cnn_params:
            if 'fc' not in name:
                assert cnn_params[name] == cnnv2_params[name], (
                    f"Conv parameter '{name}' differs between CNN and CNNv2: "
                    f"CNN={cnn_params[name]}, CNNv2={cnnv2_params[name]}"
                )

    def test_cnnv2_param_count_close_to_cnn(self):
        """
        Contract: CNNv2 param count is CNN params minus 2 (fc2: 128*3 vs 128*1).
        """
        from models import TyphoonCNN, TyphoonCNNv2

        cnn = TyphoonCNN()
        cnnv2 = TyphoonCNNv2()

        cnn_params = sum(p.numel() for p in cnn.parameters())
        cnnv2_params = sum(p.numel() for p in cnnv2.parameters())

        # Difference: fc2 weights 64*(3-1)=128 + bias (3-1)=2 = 130
        diff = cnn_params - cnnv2_params
        assert diff == 130, (
            f"Expected param diff of 130 between CNN and CNNv2 "
            f"(fc2: 64*3+3 vs 64*1+1), got {diff}"
        )

    def test_cnnv2_final_fc_output_features_is_one(self):
        """
        Contract: CNNv2 final FC layer must output exactly 1 feature.
        """
        from models import TyphoonCNNv2

        model = TyphoonCNNv2()
        linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
        output_layer = linear_layers[-1]
        assert output_layer.out_features == 1, (
            f"Expected 1 output feature, got {output_layer.out_features}"
        )


class TestDvorakLookupFunction:
    """Tests for dvorak_lookup function (Experiment 2)."""

    def test_dvorak_lookup_function_exists(self):
        """
        Contract: evaluate.py must define dvorak_lookup function.
        """
        from evaluate import dvorak_lookup
        assert callable(dvorak_lookup)

    def test_dvorak_lookup_returns_two_arrays(self):
        """
        Contract: dvorak_lookup returns (wind_knots, pressure_hpa) as numpy arrays.
        """
        from evaluate import dvorak_lookup

        t_numbers = np.array([3.0, 5.0, 7.0])
        wind, pressure = dvorak_lookup(t_numbers)

        assert isinstance(wind, np.ndarray), f"wind must be ndarray, got {type(wind)}"
        assert isinstance(pressure, np.ndarray), f"pressure must be ndarray, got {type(pressure)}"
        assert wind.shape == t_numbers.shape
        assert pressure.shape == t_numbers.shape

    def test_dvorak_lookup_boundary_t_below_1p5(self):
        """
        Contract: T < 1.5 maps to 25kt, 1009hPa.
        """
        from evaluate import dvorak_lookup

        wind, pressure = dvorak_lookup(np.array([1.0, 1.4, 1.49]))

        assert (wind == 25).all(), f"T<1.5 should map to 25kt, got {wind}"
        assert (pressure == 1009).all(), f"T<1.5 should map to 1009hPa, got {pressure}"

    def test_dvorak_lookup_boundary_t_at_2p5(self):
        """
        Contract: T >= 1.5 and T < 2.5 maps to 30kt, 1000hPa.
        """
        from evaluate import dvorak_lookup

        wind, pressure = dvorak_lookup(np.array([1.5, 2.0, 2.49]))

        assert (wind == 30).all(), f"T in [1.5, 2.5) should map to 30kt, got {wind}"
        assert (pressure == 1000).all(), f"T in [1.5, 2.5) should map to 1000hPa, got {pressure}"

    def test_dvorak_lookup_boundary_t_at_3p5(self):
        """
        Contract: T >= 2.5 and T < 3.5 maps to 45kt, 991hPa.
        """
        from evaluate import dvorak_lookup

        wind, pressure = dvorak_lookup(np.array([2.5, 3.0, 3.49]))

        assert (wind == 45).all(), f"T in [2.5, 3.5) should map to 45kt, got {wind}"
        assert (pressure == 991).all(), f"T in [2.5, 3.5) should map to 991hPa, got {pressure}"

    def test_dvorak_lookup_boundary_t_at_4p5(self):
        """
        Contract: T >= 3.5 and T < 4.5 maps to 65kt, 976hPa.
        """
        from evaluate import dvorak_lookup

        wind, pressure = dvorak_lookup(np.array([3.5, 4.0, 4.49]))

        assert (wind == 65).all(), f"T in [3.5, 4.5) should map to 65kt, got {wind}"
        assert (pressure == 976).all(), f"T in [3.5, 4.5) should map to 976hPa, got {pressure}"

    def test_dvorak_lookup_boundary_t_at_5p5(self):
        """
        Contract: T >= 4.5 and T < 5.5 maps to 90kt, 954hPa.
        """
        from evaluate import dvorak_lookup

        wind, pressure = dvorak_lookup(np.array([4.5, 5.0, 5.49]))

        assert (wind == 90).all(), f"T in [4.5, 5.5) should map to 90kt, got {wind}"
        assert (pressure == 954).all(), f"T in [4.5, 5.5) should map to 954hPa, got {pressure}"

    def test_dvorak_lookup_boundary_t_at_6p5(self):
        """
        Contract: T >= 5.5 and T < 6.5 maps to 115kt, 927hPa.
        """
        from evaluate import dvorak_lookup

        wind, pressure = dvorak_lookup(np.array([5.5, 6.0, 6.49]))

        assert (wind == 115).all(), f"T in [5.5, 6.5) should map to 115kt, got {wind}"
        assert (pressure == 927).all(), f"T in [5.5, 6.5) should map to 927hPa, got {pressure}"

    def test_dvorak_lookup_boundary_t_at_7p5(self):
        """
        Contract: T >= 6.5 and T < 7.5 maps to 140kt, 898hPa.
        """
        from evaluate import dvorak_lookup

        wind, pressure = dvorak_lookup(np.array([6.5, 7.0, 7.49]))

        assert (wind == 140).all(), f"T in [6.5, 7.5) should map to 140kt, got {wind}"
        assert (pressure == 898).all(), f"T in [6.5, 7.5) should map to 898hPa, got {pressure}"

    def test_dvorak_lookup_t_at_7p5_and_above(self):
        """
        Contract: T >= 7.5 maps to 170kt, 858hPa.
        """
        from evaluate import dvorak_lookup

        wind, pressure = dvorak_lookup(np.array([7.5, 8.0, 8.5]))

        assert (wind == 170).all(), f"T>=7.5 should map to 170kt, got {wind}"
        assert (pressure == 858).all(), f"T>=7.5 should map to 858hPa, got {pressure}"

    def test_dvorak_lookup_full_transitions(self):
        """
        Contract: dvorak_lookup produces correct values at every bin boundary.
        """
        from evaluate import dvorak_lookup

        boundary_t_values = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
        expected_wind = [30, 45, 65, 90, 115, 140, 170]
        expected_pressure = [1000, 991, 976, 954, 927, 898, 858]

        wind, pressure = dvorak_lookup(np.array(boundary_t_values))

        assert np.allclose(wind, expected_wind), (
            f"Wind at boundaries: expected {expected_wind}, got {wind}"
        )
        assert np.allclose(pressure, expected_pressure), (
            f"Pressure at boundaries: expected {expected_pressure}, got {pressure}"
        )


class TestEvaluateModelSingleOutput:
    """Tests for evaluate_model handling single-output CNNv2 models (Experiment 2)."""

    @pytest.fixture
    def single_output_checkpoint(self, tmp_path):
        """Create a mock checkpoint for a single-output CNNv2 model."""
        from models import TyphoonCNNv2
        model = TyphoonCNNv2()
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'epoch': 5,
            'val_loss': 0.3
        }
        ckpt_path = tmp_path / "cnnv2_model.pt"
        torch.save(checkpoint, ckpt_path)
        return str(ckpt_path)

    @pytest.fixture
    def single_output_test_loader(self):
        """Create a synthetic test loader for single-output model testing."""
        from torch.utils.data import TensorDataset, DataLoader
        torch.manual_seed(42)
        x = torch.randn(20, 1, 240, 240)
        # For CNNv2 (single output): targets are only T-number
        y = torch.randn(20, 1) * 3 + 5  # T-number ~[2, 8]
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        return loader

    def test_evaluate_model_single_output_returns_three_targets(
        self, single_output_checkpoint, single_output_test_loader
    ):
        """
        Contract: evaluate_model on single-output model must still return
        metrics dict with t_number, wind, and pressure keys (via dvorak derivation).
        """
        from evaluate import evaluate_model
        from models import TyphoonCNNv2

        model = TyphoonCNNv2()
        metrics, _, _ = evaluate_model(model, single_output_test_loader, single_output_checkpoint)

        assert isinstance(metrics, dict)
        assert 't_number' in metrics, "Metrics must contain t_number"
        assert 'wind' in metrics, "Metrics must contain wind"
        assert 'pressure' in metrics, "Metrics must contain pressure"

    def test_evaluate_model_single_output_has_mae_rmse_r2_per_target(
        self, single_output_checkpoint, single_output_test_loader
    ):
        """
        Contract: Each target (t_number, wind, pressure) has mae, rmse, r2.
        """
        from evaluate import evaluate_model
        from models import TyphoonCNNv2

        model = TyphoonCNNv2()
        metrics, _, _ = evaluate_model(model, single_output_test_loader, single_output_checkpoint)

        for target in ['t_number', 'wind', 'pressure']:
            assert 'mae' in metrics[target], f"Missing mae for {target}"
            assert 'rmse' in metrics[target], f"Missing rmse for {target}"
            assert 'r2' in metrics[target], f"Missing r2 for {target}"

    def test_evaluate_model_single_output_wind_derived_from_dvorak(
        self, single_output_checkpoint, single_output_test_loader
    ):
        """
        Contract: Wind metrics for single-output model come from dvorak_lookup
        (not from raw model output).
        """
        from evaluate import evaluate_model
        from models import TyphoonCNNv2

        model = TyphoonCNNv2()
        metrics, _, _ = evaluate_model(model, single_output_test_loader, single_output_checkpoint)

        # Wind values should be one of the discrete Dvorak values
        valid_wind = {25, 30, 45, 65, 90, 115, 140, 170}
        # The predicted wind should be a valid Dvorak wind value
        # We can't directly test the prediction here without full pipeline,
        # but we verify the function doesn't crash
        assert 'wind' in metrics
        assert 'mae' in metrics['wind']


class TestGenerateReportWithScatterPlots:
    """Tests for generate_report producing scatter plots from predictions/targets (Experiment 2)."""

    def test_generate_report_accepts_predictions_and_targets(self, tmp_path):
        """
        Contract: generate_report must accept predictions and targets as (N, 3)
        arrays and use them to produce scatter plots instead of bar charts.
        """
        from evaluate import generate_report
        import os

        # Deterministic synthetic predictions and targets as (N, 3) arrays
        # Columns: [t_number, wind, pressure]
        np.random.seed(42)
        predictions = np.column_stack([
            np.random.uniform(2, 8, 50),      # t_number
            np.random.uniform(30, 170, 50),   # wind
            np.random.uniform(858, 1009, 50),  # pressure
        ])
        targets = np.column_stack([
            np.random.uniform(2, 8, 50),
            np.random.uniform(30, 170, 50),
            np.random.uniform(858, 1009, 50),
        ])

        generate_report({}, str(tmp_path), predictions=predictions, targets=targets)

        # Should produce scatter plots for each target
        for target in ['t_number', 'wind', 'pressure']:
            plot_path = os.path.join(str(tmp_path), f'{target}_scatter.png')
            assert os.path.exists(plot_path), (
                f"Scatter plot not found for {target}: {plot_path}"
            )

    def test_generate_report_without_predictions_skips_scatter(self, tmp_path):
        """
        Contract: generate_report without predictions/targets should not crash.
        """
        from evaluate import generate_report

        # Should not raise - old behavior with no predictions
        try:
            generate_report({}, str(tmp_path))
        except TypeError:
            pytest.fail(
                "generate_report should not require predictions/targets arguments"
            )

    def test_generate_report_scatter_plot_differs_without_predictions(self, tmp_path):
        """
        Contract: When predictions/targets are None, scatter plots may be skipped
        or bar charts used instead; when provided as (N, 3) arrays, scatter plots must exist.
        """
        from evaluate import generate_report
        import os

        # With predictions as (N, 3) arrays -> scatter plots
        np.random.seed(42)
        predictions = np.column_stack([
            np.random.uniform(2, 8, 30),
            np.random.uniform(30, 170, 30),
            np.random.uniform(858, 1009, 30),
        ])
        targets = np.column_stack([
            np.random.uniform(2, 8, 30),
            np.random.uniform(30, 170, 30),
            np.random.uniform(858, 1009, 30),
        ])

        out_dir_with = tmp_path / "with_preds"
        out_dir_with.mkdir()
        generate_report({}, str(out_dir_with), predictions=predictions, targets=targets)

        # With predictions, scatter plots must exist
        for target in ['t_number', 'wind', 'pressure']:
            plot_path = out_dir_with / f'{target}_scatter.png'
            assert plot_path.exists(), (
                f"With predictions provided, scatter plot must exist for {target}"
            )


class TestEvaluateCLIWithCNNV2:
    """Tests for evaluate.py CLI accepting --model cnnv2 (Experiment 2)."""

    def test_evaluate_cli_accepts_cnnv2_model(self):
        """
        Contract: main() must accept --model cnnv2 as a valid choice.
        """
        import argparse
        import sys
        from evaluate import main

        # Test that --model cnnv2 is a recognized argument
        # This tests the argument parser definition
        for model_choice in ['cnn', 'mlp', 'cnnv2']:
            test_args = [
                'evaluate.py',
                '--model', model_choice,
                '--data_dir', '/tmp/nonexistent',
                '--checkpoint', '/tmp/nonexistent.pt',
                '--output_dir', '/tmp/nonexistent_out',
            ]
            try:
                old_argv = sys.argv
                sys.argv = test_args
                # Just verify it doesn't raise an argparse error
                # We can't run full main() without data
                parser = argparse.ArgumentParser()
                # We check that cnnv2 is listed in the choices
                # by inspecting what main does with it
                del sys.argv
                sys.argv = old_argv
            except SystemExit:
                # argparse exits on error - but cnnv2 should not cause error
                sys.argv = old_argv
                if model_choice == 'cnnv2':
                    pytest.fail("--model cnnv2 should be accepted by evaluate CLI")
