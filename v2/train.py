"""
Training loop for storm bounding box regression.

Phased unfreezing schedule for pretrained ResNet18 backbone:
  Phase 1 (ep 1-5):   backbone frozen, head trains at lr
  Phase 2 (ep 6-20):  layer3+layer4 unfrozen at lr*0.1
  Phase 3 (ep 21+):   all unfrozen at lr*0.01

Loss options:
  smoothl1: coordinate-wise SmoothL1 (baseline)
  giou: Generalized IoU loss (directly optimizes overlap)
  diou: Distance-IoU (penalizes center distance, targets position error)

Checkpoint resume via --resume (saves latest.pt every epoch).
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

# Phase transition epochs
PHASE2_EPOCH = 5
PHASE3_EPOCH = 20


def giou_loss(pred, target):
    """Generalized IoU loss for bounding box regression.

    Args:
        pred: (N, 4) predicted boxes (minX, minY, maxX, maxY) in [0, 1]
        target: (N, 4) target boxes

    Returns:
        Scalar loss (1 - mean GIoU), range [0, 2]
    """
    # Intersection
    inter_min = torch.max(pred[:, :2], target[:, :2])
    inter_max = torch.min(pred[:, 2:], target[:, 2:])
    inter_wh = (inter_max - inter_min).clamp(min=0)
    inter_area = inter_wh[:, 0] * inter_wh[:, 1]

    # Areas
    pred_wh = (pred[:, 2:] - pred[:, :2]).clamp(min=0)
    target_wh = (target[:, 2:] - target[:, :2]).clamp(min=0)
    pred_area = pred_wh[:, 0] * pred_wh[:, 1]
    target_area = target_wh[:, 0] * target_wh[:, 1]
    union_area = pred_area + target_area - inter_area

    iou = inter_area / (union_area + 1e-7)

    # Enclosing box
    encl_min = torch.min(pred[:, :2], target[:, :2])
    encl_max = torch.max(pred[:, 2:], target[:, 2:])
    encl_wh = (encl_max - encl_min).clamp(min=0)
    encl_area = encl_wh[:, 0] * encl_wh[:, 1]

    giou = iou - (encl_area - union_area) / (encl_area + 1e-7)
    return (1 - giou).mean()


def diou_loss(pred, target):
    """Distance-IoU loss for bounding box regression.

    DIoU = IoU - rho^2 / c^2, where rho is the Euclidean distance between
    predicted and target box centers, and c is the diagonal length of the
    smallest enclosing box. Directly penalizes center-to-center displacement,
    which coordinate-wise SmoothL1 does not capture.

    Args:
        pred: (N, 4) predicted boxes (minX, minY, maxX, maxY) in [0, 1]
        target: (N, 4) target boxes

    Returns:
        Scalar loss (1 - mean DIoU)
    """
    # Intersection
    inter_min = torch.max(pred[:, :2], target[:, :2])
    inter_max = torch.min(pred[:, 2:], target[:, 2:])
    inter_wh = (inter_max - inter_min).clamp(min=0)
    inter_area = inter_wh[:, 0] * inter_wh[:, 1]

    pred_wh = (pred[:, 2:] - pred[:, :2]).clamp(min=0)
    target_wh = (target[:, 2:] - target[:, :2]).clamp(min=0)
    pred_area = pred_wh[:, 0] * pred_wh[:, 1]
    target_area = target_wh[:, 0] * target_wh[:, 1]
    union_area = pred_area + target_area - inter_area
    iou = inter_area / (union_area + 1e-7)

    # Center distance squared
    pred_cx = (pred[:, 0] + pred[:, 2]) / 2
    pred_cy = (pred[:, 1] + pred[:, 3]) / 2
    target_cx = (target[:, 0] + target[:, 2]) / 2
    target_cy = (target[:, 1] + target[:, 3]) / 2
    center_dist_sq = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2

    # Enclosing-box diagonal squared
    encl_min = torch.min(pred[:, :2], target[:, :2])
    encl_max = torch.max(pred[:, 2:], target[:, 2:])
    encl_wh = (encl_max - encl_min).clamp(min=0)
    encl_diag_sq = encl_wh[:, 0] ** 2 + encl_wh[:, 1] ** 2

    diou = iou - center_dist_sq / (encl_diag_sq + 1e-7)
    return (1 - diou).mean()


def _build_optimizer(model, lr, phase):
    """Build Adam optimizer with per-group learning rates for current phase."""
    head_params = list(model.head.parameters())
    backbone_params = [p for n, p in model.named_parameters()
                       if not n.startswith('head.') and p.requires_grad]

    if phase == 1:
        return optim.Adam(head_params, lr=lr)
    elif phase == 2:
        return optim.Adam([
            {'params': backbone_params, 'lr': lr * 0.1},
            {'params': head_params, 'lr': lr},
        ])
    else:
        return optim.Adam([
            {'params': backbone_params, 'lr': lr * 0.01},
            {'params': head_params, 'lr': lr},
        ])


def train_bbox_model(model, train_loader, val_loader, args):
    """Train bbox regression model with phased unfreezing.

    Args:
        model: StormBboxNet instance
        train_loader: Training data loader
        val_loader: Validation data loader
        args: Object with: checkpoint_dir, epochs, lr
              Optional: resume (bool)

    Returns:
        dict: {train_losses, val_losses, best_epoch}
    """
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    model = model.to(device)

    loss_type = getattr(args, 'loss', 'smoothl1')
    smooth_l1 = nn.SmoothL1Loss()
    if loss_type == 'giou':
        def criterion(pred, target):
            return smooth_l1(pred, target) + giou_loss(pred, target)
    elif loss_type == 'diou':
        def criterion(pred, target):
            return smooth_l1(pred, target) + diou_loss(pred, target)
    elif loss_type == 'cxywh':
        def criterion(pred, target):
            # SmoothL1 on minmax (position accuracy) + SmoothL1 on cxywh (size signal)
            def to_cxywh(boxes):
                cx = (boxes[:, 0] + boxes[:, 2]) / 2
                cy = (boxes[:, 1] + boxes[:, 3]) / 2
                w = boxes[:, 2] - boxes[:, 0]
                h = boxes[:, 3] - boxes[:, 1]
                return torch.stack([cx, cy, w, h], dim=1)
            return smooth_l1(pred, target) + smooth_l1(to_cxywh(pred), to_cxywh(target))
    else:
        criterion = smooth_l1

    # Start in phase 1
    model.freeze_backbone()
    current_phase = 1
    optimizer = _build_optimizer(model, args.lr, current_phase)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

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
            best_val_loss = ckpt['best_val_loss']
            patience_counter = ckpt['patience_counter']
            train_losses = ckpt['train_losses']
            val_losses = ckpt['val_losses']
            best_epoch = ckpt['best_epoch']
            start_epoch = ckpt['epoch'] + 1
            current_phase = ckpt.get('phase', 1)

            # Restore phase state
            if current_phase >= 3:
                model.unfreeze_all()
            elif current_phase >= 2:
                model.unfreeze_top_blocks()

            optimizer = _build_optimizer(model, args.lr, current_phase)
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5
            )
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])

            print(f"Resumed from epoch {start_epoch} (phase {current_phase}, "
                  f"best val_loss: {best_val_loss:.4f} at epoch {best_epoch})")
        except Exception as e:
            print(f"Warning: could not load latest.pt ({e}), starting fresh")
            start_epoch = 0

    for epoch in range(start_epoch, args.epochs):
        # Phase transitions
        new_phase = current_phase
        if epoch == PHASE2_EPOCH and current_phase < 2:
            new_phase = 2
            model.unfreeze_top_blocks()
            print(f"Phase 2: unfreezing layer3 + layer4")
        elif epoch == PHASE3_EPOCH and current_phase < 3:
            new_phase = 3
            model.unfreeze_all()
            print(f"Phase 3: unfreezing all layers")

        if new_phase != current_phase:
            current_phase = new_phase
            optimizer = _build_optimizer(model, args.lr, current_phase)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5
            )

        # Training
        model.train()
        running_loss = 0.0
        n_batches = 0

        batch_iter = train_loader
        if tqdm is not None:
            batch_iter = tqdm(train_loader, desc=f"Epoch {epoch+1:3d}/{args.epochs}",
                              leave=False, unit="batch")

        mixup_alpha = float(getattr(args, 'mixup_alpha', 0.0) or 0.0)

        for batch_x, batch_y in batch_iter:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Mixup: blend each sample with a random permutation of the batch.
            # Labels are blended linearly as well — bbox regression targets are
            # numeric, so a weighted average is a valid soft target.
            if mixup_alpha > 0.0 and batch_x.size(0) > 1:
                lam = float(torch.distributions.Beta(mixup_alpha, mixup_alpha).sample())
                perm = torch.randperm(batch_x.size(0), device=device)
                batch_x = lam * batch_x + (1.0 - lam) * batch_x[perm]
                batch_y = lam * batch_y + (1.0 - lam) * batch_y[perm]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        epoch_train_loss = running_loss / n_batches
        train_losses.append(epoch_train_loss)

        # Validation
        epoch_val_loss = float('inf')
        if val_loader is not None and len(val_loader) > 0:
            model.eval()
            running_val = 0.0
            n_val = 0

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    running_val += loss.item()
                    n_val += 1

            epoch_val_loss = running_val / n_val
            val_losses.append(epoch_val_loss)
            scheduler.step(epoch_val_loss)

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                patience_counter = 0
                best_epoch = epoch
                torch.save(
                    {'model_state_dict': model.state_dict(), 'epoch': epoch,
                     'val_loss': best_val_loss},
                    os.path.join(args.checkpoint_dir, 'best_model.pt')
                )
            else:
                patience_counter += 1
        else:
            val_losses.append(float('inf'))

        # Save latest checkpoint
        torch.save({
            'epoch': epoch,
            'phase': current_phase,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_epoch': best_epoch,
        }, latest_path)

        lr_str = optimizer.param_groups[0]['lr']
        val_str = f"{epoch_val_loss:.4f}" if epoch_val_loss < float('inf') else "n/a"
        print(f"Epoch {epoch+1:3d}/{args.epochs} | phase {current_phase} | "
              f"train: {epoch_train_loss:.4f} | val: {val_str} | "
              f"best: {best_val_loss:.4f} (ep {best_epoch+1}) | "
              f"lr: {lr_str:.1e} | patience: {patience_counter}/{early_stopping_patience}")

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    return {'train_losses': train_losses, 'val_losses': val_losses, 'best_epoch': best_epoch}


def main():
    parser = argparse.ArgumentParser(description='Train Storm Bbox Model')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--loss', type=str, default='smoothl1',
                        choices=['smoothl1', 'giou', 'diou', 'cxywh'],
                        help='Loss function: smoothl1, giou (SmoothL1+GIoU), '
                             'diou (SmoothL1+DIoU), or cxywh (SmoothL1 in center/size space)')
    parser.add_argument('--no-augment', action='store_true',
                        help='Disable train-time augmentation')
    parser.add_argument('--mixup_alpha', type=float, default=0.0,
                        help='Mixup Beta(a,a) parameter. 0 = disabled (default). '
                             'Blends image+label pairs within batch.')
    parser.add_argument('--resume', action='store_true')

    args = parser.parse_args()

    from v2.models import StormBboxNet
    from v2.dataset import create_bbox_dataloaders

    model = StormBboxNet(pretrained=True)
    augment = not getattr(args, 'no_augment', False)
    train_loader, val_loader, _ = create_bbox_dataloaders(
        args.data_dir, batch_size=args.batch_size, augment=augment)

    history = train_bbox_model(model, train_loader, val_loader, args)
    print(f"Training complete. Best epoch: {history['best_epoch'] + 1}")


if __name__ == '__main__':
    main()
