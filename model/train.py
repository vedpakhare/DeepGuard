"""
DeepGuard - training script.
Fine-tunes a pretrained EfficientNet-B0 to classify face images as real or fake.
Run on Google Colab with a T4 GPU.

Example:
    python model/train.py --root data/real_and_fake_face --real training_real --fake training_fake
"""

import argparse
import os

import timm
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from dataset import get_loaders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/real_and_fake_face")
    parser.add_argument("--real", default="training_real", help="name of the real-images folder")
    parser.add_argument("--fake", default="training_fake", help="name of the fake-images folder")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", default=os.path.join("model", "weights", "detector.pth"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader, val_loader, full_ds = get_loaders(
        args.root, args.real, args.fake, batch_size=args.batch_size)
    real, fake = full_ds.counts()
    print(f"Total: {len(full_ds)} | real: {real} | fake: {fake}")

    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            loss = loss_fn(model(images), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                preds += model(images.to(device)).argmax(1).cpu().tolist()
                truths += labels.tolist()

        acc = accuracy_score(truths, preds)
        print(f"Epoch {epoch + 1}: loss={total_loss:.2f} | acc={acc:.2%} | "
              f"prec={precision_score(truths, preds):.2f} "
              f"rec={recall_score(truths, preds):.2f} f1={f1_score(truths, preds):.2f}")

        # keep the best epoch, not just the last one
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), args.out)
            print("  saved best checkpoint")

    print(f"Done. Best validation accuracy: {best_acc:.2%} -> {args.out}")


if __name__ == "__main__":
    main()
