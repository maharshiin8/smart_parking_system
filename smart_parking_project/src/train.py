"""
train.py
--------
Trains the ParkingSpotClassifier and saves the best checkpoint
(by validation accuracy) to models/best_model.pt.

Usage:
    python src/train.py --epochs 8 --batch_size 32 --lr 1e-4
"""
import argparse
import os
import time
import json

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from model import build_model


def evaluate(model, loader, device, criterion):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            loss_sum += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())
    return loss_sum / total, correct / total, all_preds, all_labels


def confusion_matrix(preds, labels, num_classes=2):
    cm = [[0] * num_classes for _ in range(num_classes)]
    for p, l in zip(preds, labels):
        cm[l][p] += 1
    return cm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/dataset")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out_dir", default="models")
    parser.add_argument("--freeze_backbone", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    train_loader, val_loader, classes = get_dataloaders(args.data_root, args.batch_size)
    print(f"Classes: {classes} (index order matters: 0={classes[0]}, 1={classes[1]})")

    model = build_model(device=device, pretrained=True, freeze_backbone=args.freeze_backbone)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    history = []
    best_acc = 0.0
    best_path = os.path.join(args.out_dir, "best_model.pt")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            running_correct += (out.argmax(dim=1) == y).sum().item()
            running_total += y.size(0)

        scheduler.step()
        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, device, criterion)

        dt = time.time() - t0
        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | {dt:.1f}s")

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        })

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "classes": classes,
                "val_acc": val_acc,
                "epoch": epoch,
            }, best_path)
            print(f"  -> new best model saved (val_acc={val_acc:.4f})")

    cm = confusion_matrix(val_preds, val_labels)
    print("Final confusion matrix (rows=true, cols=pred):", cm)

    with open(os.path.join(args.out_dir, "training_history.json"), "w") as f:
        json.dump({"history": history, "best_val_acc": best_acc, "confusion_matrix": cm,
                    "classes": classes}, f, indent=2)

    print(f"\nBest validation accuracy: {best_acc:.4f}")
    print(f"Best model saved to: {best_path}")


if __name__ == "__main__":
    main()
