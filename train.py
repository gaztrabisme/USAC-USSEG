"""
Typhoon Intensity Estimation ML Pipeline - Training Module

Supports checkpoint resume: saves latest.pt every epoch with full state
(optimizer, scheduler, patience, losses). Use --resume to continue training.
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


class WeightedMSELoss(nn.Module):
    """Weighted MSE loss with target scaling.

    Scales targets by dividing: t_number/8, wind/170, pressure/1009
    Then compute MSE between scaled predictions and scaled targets.
    """

    def __init__(self):
        super(WeightedMSELoss, self).__init__()
        self.register_buffer('scales', torch.tensor([8.0, 170.0, 1009.0]))

    def forward(self, pred, target):
        scaled_target = target / self.scales
        scaled_pred = pred / self.scales
        loss = nn.functional.mse_loss(scaled_pred, scaled_target)
        return loss


class TNumberMSELoss(nn.Module):
    """MSE loss on T-number only, scaled by 1/8."""

    def forward(self, pred, target):
        scaled_pred = pred / 8.0
        scaled_target = target / 8.0
        return nn.functional.mse_loss(scaled_pred, scaled_target)


class BoundaryAwareLoss(nn.Module):
    """MSE on T-number + differentiable Dvorak wind penalty.

    Uses steep sigmoid approximations of the Dvorak step function so that
    boundary crossings produce gradient signal proportional to the wind
    jump at that boundary.

    loss = MSE(pred_t/8, true_t/8) + λ * MSE(soft_wind(pred_t)/170, soft_wind(true_t)/170)
    """

    def __init__(self, wind_lambda=1.0, steepness=10.0):
        super().__init__()
        self.wind_lambda = wind_lambda
        self.steepness = steepness
        # Dvorak thresholds and cumulative wind increments
        self.register_buffer('thresholds',
                             torch.tensor([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]))
        self.register_buffer('wind_jumps',
                             torch.tensor([5.0, 15.0, 20.0, 25.0, 25.0, 25.0, 30.0]))
        self.base_wind = 25.0

    def _soft_dvorak_wind(self, t):
        """Differentiable approximation of Dvorak wind lookup.

        At each threshold, adds wind_jump * sigmoid(steepness * (t - threshold)).
        Sharp sigmoids approximate the step function while remaining differentiable.
        """
        # t: (N, 1) or (N,)
        t_flat = t.view(-1, 1)  # (N, 1)
        # (N, 7) — how far past each threshold
        diffs = self.steepness * (t_flat - self.thresholds.unsqueeze(0))
        steps = torch.sigmoid(diffs)  # (N, 7)
        wind = self.base_wind + (steps * self.wind_jumps.unsqueeze(0)).sum(dim=1)
        return wind  # (N,)

    def forward(self, pred, target):
        # T-number MSE (same as TNumberMSELoss)
        t_loss = nn.functional.mse_loss(pred / 8.0, target / 8.0)

        # Wind-space penalty via differentiable Dvorak
        pred_wind = self._soft_dvorak_wind(pred)
        true_wind = self._soft_dvorak_wind(target)
        wind_loss = nn.functional.mse_loss(pred_wind / 170.0, true_wind / 170.0)

        return t_loss + self.wind_lambda * wind_loss


def train_model(model, train_loader, val_loader, args):
    """Train the model and return training history.

    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        args: Object with attributes: checkpoint_dir, epochs, batch_size, lr
              Optional: resume (bool) — load latest.pt and continue
              Optional: single_target (bool) — use T-number only loss

    Returns:
        dict: {train_losses: list[float], val_losses: list[float], best_epoch: int}
    """
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    model = model.to(device)

    single_target = getattr(args, 'single_target', False)
    boundary_aware = getattr(args, 'boundary_aware', False)
    if boundary_aware:
        criterion = BoundaryAwareLoss(
            wind_lambda=getattr(args, 'wind_lambda', 1.0),
            steepness=getattr(args, 'boundary_steepness', 10.0),
        ).to(device)
    elif single_target:
        criterion = TNumberMSELoss().to(device)
    else:
        criterion = WeightedMSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training state
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 15
    train_losses = []
    val_losses = []
    best_epoch = 0
    start_epoch = 0

    # Resume from checkpoint
    resume = getattr(args, 'resume', False)
    latest_path = os.path.join(args.checkpoint_dir, 'latest.pt')

    if resume and os.path.exists(latest_path):
        try:
            ckpt = torch.load(latest_path, weights_only=False, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            best_val_loss = ckpt['best_val_loss']
            patience_counter = ckpt['patience_counter']
            train_losses = ckpt['train_losses']
            val_losses = ckpt['val_losses']
            best_epoch = ckpt['best_epoch']
            start_epoch = ckpt['epoch'] + 1
            print(f"Resumed from epoch {start_epoch} (best val_loss: {best_val_loss:.4f} at epoch {best_epoch})")
        except Exception as e:
            print(f"Warning: could not load latest.pt ({e}), starting fresh")
            start_epoch = 0

    for epoch in range(start_epoch, args.epochs):
        # Training phase
        model.train()
        running_train_loss = 0.0
        num_train_batches = 0

        batch_iter = train_loader
        if tqdm is not None:
            batch_iter = tqdm(train_loader, desc=f"Epoch {epoch+1:3d}/{args.epochs}",
                              leave=False, unit="batch")

        for batch_x, batch_y in batch_iter:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            target = batch_y[:, 0:1] if single_target else batch_y
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            num_train_batches += 1

        epoch_train_loss = running_train_loss / num_train_batches
        train_losses.append(epoch_train_loss)

        # Validation phase
        epoch_val_loss = float('inf')
        val_loader_is_valid = val_loader is not None and len(val_loader) > 0

        if val_loader_is_valid:
            model.eval()
            running_val_loss = 0.0
            num_val_batches = 0

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    outputs = model(batch_x)
                    target = batch_y[:, 0:1] if single_target else batch_y
                    loss = criterion(outputs, target)
                    running_val_loss += loss.item()
                    num_val_batches += 1

            epoch_val_loss = running_val_loss / num_val_batches
            val_losses.append(epoch_val_loss)

            scheduler.step(epoch_val_loss)

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                patience_counter = 0
                best_epoch = epoch

                # Save best model (for inference)
                best_checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'val_loss': best_val_loss
                }
                torch.save(best_checkpoint,
                           os.path.join(args.checkpoint_dir, 'best_model.pt'))
            else:
                patience_counter += 1
        else:
            val_losses.append(float('inf'))
            if epoch == 0 or epoch_train_loss < best_val_loss:
                best_val_loss = epoch_train_loss
                best_epoch = epoch
                best_checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'val_loss': best_val_loss
                }
                torch.save(best_checkpoint,
                           os.path.join(args.checkpoint_dir, 'best_model.pt'))

        # Save latest checkpoint (for resume)
        latest_checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_epoch': best_epoch,
        }
        torch.save(latest_checkpoint, latest_path)

        # Epoch progress line
        lr = optimizer.param_groups[0]['lr']
        val_str = f"{epoch_val_loss:.4f}" if val_loader_is_valid else "n/a"
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"train: {epoch_train_loss:.4f} | val: {val_str} | "
              f"best: {best_val_loss:.4f} (ep {best_epoch+1}) | "
              f"lr: {lr:.1e} | patience: {patience_counter}/{early_stopping_patience}")

        # Early stopping
        if val_loader_is_valid and patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_epoch': best_epoch
    }


def main():
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(description='Train Typhoon Intensity Estimation Model')
    parser.add_argument('--model', type=str, required=True, choices=['mlp', 'cnn', 'cnnv2'],
                        help='Model type: mlp, cnn, or cnnv2')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to training data directory')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Directory to save checkpoints')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from latest checkpoint')

    args = parser.parse_args()

    from models import TyphoonMLP, TyphoonCNN, TyphoonCNNv2
    from dataset import create_dataloaders, create_balanced_dataloaders

    if args.model == 'mlp':
        model = TyphoonMLP()
    elif args.model == 'cnn':
        model = TyphoonCNN()
    else:
        model = TyphoonCNNv2()
        args.single_target = True

    if args.model == 'cnnv2':
        train_loader, val_loader, _ = create_balanced_dataloaders(
            args.data_dir, batch_size=args.batch_size)
    else:
        train_loader, val_loader, _ = create_dataloaders(args.data_dir, batch_size=args.batch_size)

    history = train_model(model, train_loader, val_loader, args)
    print(f"Training complete. Best epoch: {history['best_epoch'] + 1}")
    print(f"Final train loss: {history['train_losses'][-1]:.4f}")


if __name__ == '__main__':
    main()
