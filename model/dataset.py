"""
DeepGuard - dataset loading shared by train.py and evaluate.py.
Expects a folder with one sub-folder of real images and one of fake images.
Labels: 0 = real, 1 = fake.
"""

import os
import random

from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# Training images get random flips/rotations so the model can't just memorise them.
TRAIN_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    NORM,
])

# Validation / test images are left untouched so the scores are honest.
EVAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    NORM,
])


class FaceDataset(Dataset):
    def __init__(self, root, real_folder, fake_folder, transform):
        self.samples = []
        for folder, label in {real_folder: 0, fake_folder: 1}.items():
            path = os.path.join(root, folder)
            for fname in sorted(os.listdir(path)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(path, fname), label))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label

    def counts(self):
        real = sum(1 for _, label in self.samples if label == 0)
        return real, len(self.samples) - real


def get_loaders(root, real_folder, fake_folder, batch_size=32, seed=42, num_workers=2):
    """80/20 train/validation split. The seed is fixed so the split is the same every run."""
    train_ds = FaceDataset(root, real_folder, fake_folder, TRAIN_TF)
    val_ds = FaceDataset(root, real_folder, fake_folder, EVAL_TF)

    idx = list(range(len(train_ds)))
    random.Random(seed).shuffle(idx)
    cut = int(0.8 * len(idx))

    train_loader = DataLoader(Subset(train_ds, idx[:cut]), batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(Subset(val_ds, idx[cut:]), batch_size=batch_size,
                            num_workers=num_workers)
    return train_loader, val_loader, train_ds


def get_full_eval_loader(root, real_folder, fake_folder, batch_size=32, num_workers=2):
    """Every image in the folder, no augmentation. Used for cross-dataset testing."""
    ds = FaceDataset(root, real_folder, fake_folder, EVAL_TF)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers), ds
