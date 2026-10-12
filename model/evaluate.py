"""
DeepGuard - evaluation script.

1) Held-out 20% of the training dataset (same split as train.py):
    python model/evaluate.py --root data/real_and_fake_face

2) Cross-dataset test on a dataset the model never trained on (use --all):
    python model/evaluate.py --root data/other_dataset --real real --fake fake --all

The drop in score between 1) and 2) is the generalisation gap we report.
"""

import argparse
import os

import timm
import torch
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

from dataset import get_loaders, get_full_eval_loader


def evaluate(model, loader, device):
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for images, labels in loader:
            preds += model(images.to(device)).argmax(1).cpu().tolist()
            truths += labels.tolist()
    return {
        "accuracy": accuracy_score(truths, preds),
        "precision": precision_score(truths, preds),
        "recall": recall_score(truths, preds),
        "f1": f1_score(truths, preds),
        "confusion_matrix": confusion_matrix(truths, preds).tolist(),  # [[TN, FP], [FN, TP]]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--real", default="training_real")
    parser.add_argument("--fake", default="training_fake")
    parser.add_argument("--weights", default=os.path.join("model", "weights", "detector.pth"))
    parser.add_argument("--all", action="store_true",
                        help="evaluate on every image (use for a dataset the model never saw)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)

    if args.all:
        loader, ds = get_full_eval_loader(args.root, args.real, args.fake)
        print(f"Evaluating on ALL {len(ds)} images (cross-dataset test)")
    else:
        _, loader, _ = get_loaders(args.root, args.real, args.fake)
        print("Evaluating on the held-out 20% validation split")

    results = evaluate(model, loader, device)
    for name, value in results.items():
        if name == "confusion_matrix":
            print(f"confusion matrix [[TN, FP], [FN, TP]]: {value}")
        else:
            print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
