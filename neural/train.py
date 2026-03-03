#!/usr/bin/env python3
"""
Training script for Frequency Recovery Network (FRNet).

Usage:
    python train.py --image-dir /path/to/images --epochs 100

Or with pre-computed data:
    python train.py --data-dir /path/to/precomputed --epochs 100
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from model import FRNet, FRNetLite
from dataset import WatermarkDataset, PrecomputedDataset


def watermark_region_loss(pred, target, p):
    """
    MSE loss on the watermark region.

    Both pred and target are (B, 2, p, p) where channels are [real, imag].
    """
    return nn.functional.mse_loss(pred, target)


def train_epoch(model, loader, optimizer, device, p):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for attacked, target, wallets in tqdm(loader, desc="Training", leave=False):
        attacked = attacked.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        pred = model(attacked)
        loss = watermark_region_loss(pred, target, p)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def validate(model, loader, device, p):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for attacked, target, wallets in tqdm(loader, desc="Validating", leave=False):
            attacked = attacked.to(device)
            target = target.to(device)

            pred = model(attacked)
            loss = watermark_region_loss(pred, target, p)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description="Train FRNet for watermark recovery")
    parser.add_argument("--image-dir", type=Path, help="Directory with training images")
    parser.add_argument("--data-dir", type=Path, help="Directory with pre-computed data")
    parser.add_argument("--output-dir", type=Path, default=Path("weights"), help="Output directory")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--image-size", type=int, default=512, help="Image size")
    parser.add_argument("--lite", action="store_true", help="Use lightweight model")
    parser.add_argument("--strength", type=float, default=2.0, help="Watermark strength")
    parser.add_argument("--resume", type=Path, help="Resume from checkpoint")
    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Calculate prime for image size
    def largest_prime(n):
        for p in range(n, 1, -1):
            if all(p % i != 0 for i in range(2, int(p**0.5) + 1)):
                return p
        return 2

    p = largest_prime(args.image_size - 2)
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Prime p: {p}")

    # Create dataset
    if args.data_dir:
        dataset = PrecomputedDataset(
            data_dir=args.data_dir,
            p=p,
        )
    elif args.image_dir:
        embed_binary = Path(__file__).parent.parent / "target/release/embed"
        if not embed_binary.exists():
            print(f"Error: {embed_binary} not found. Run 'cargo build --release' first.")
            return

        dataset = WatermarkDataset(
            image_dir=args.image_dir,
            embed_binary=embed_binary,
            image_size=args.image_size,
            p=p,
            strength=args.strength,
        )
    else:
        print("Error: Specify --image-dir or --data-dir")
        return

    # Split dataset
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Create model
    if args.lite:
        model = FRNetLite(p=p)
        print("Using FRNetLite")
    else:
        model = FRNet(p=p)
        print("Using FRNet")

    print(f"Parameters: {model.count_parameters():,}")
    model = model.to(device)

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}")

    # Output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = {
        "image_size": args.image_size,
        "p": p,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "lite": args.lite,
        "strength": args.strength,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Output: {run_dir}\n")

    history = {"train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, p)
        history["train_loss"].append(train_loss)

        # Validate
        val_loss = validate(model, val_loader, device, p)
        history["val_loss"].append(val_loss)

        # Learning rate
        lr = optimizer.param_groups[0]["lr"]
        history["lr"].append(lr)
        scheduler.step()

        # Print progress
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"LR: {lr:.2e}")

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "config": config,
        }

        # Save latest
        torch.save(checkpoint, run_dir / "latest.pth")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, run_dir / "best.pth")
            print(f"  ✓ New best model saved (val_loss: {val_loss:.6f})")

        # Save history
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))

    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {run_dir / 'best.pth'}")


if __name__ == "__main__":
    main()
