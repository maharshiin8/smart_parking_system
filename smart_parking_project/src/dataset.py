"""
dataset.py
----------
Builds PyTorch DataLoaders from an ImageFolder-style directory:

    data/dataset/train/empty/*.jpg
    data/dataset/train/occupied/*.jpg
    data/dataset/val/empty/*.jpg
    data/dataset/val/occupied/*.jpg

Works with either the synthetic demo dataset or a real dataset
(PKLot / CNRPark-EXT) reorganized into this same layout.
"""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

IMG_SIZE = 64

# ImageNet normalization stats (required because the backbone is
# pretrained on ImageNet)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def get_dataloaders(data_root="data/dataset", batch_size=32, num_workers=2):
    train_ds = datasets.ImageFolder(f"{data_root}/train", transform=train_transform)
    val_ds = datasets.ImageFolder(f"{data_root}/val", transform=eval_transform)

    # sanity: class_to_idx should be {'empty': 0, 'occupied': 1}
    assert train_ds.classes == val_ds.classes, "train/val class folders differ"

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, train_ds.classes
