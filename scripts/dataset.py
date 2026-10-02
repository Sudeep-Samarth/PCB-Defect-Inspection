# Step2
'''
1. Split the data into Train and Test sets
2. Apply Data Augmentation
3. Load the data using DataLoader
'''

import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ImageNet normalization statistics used for standard transfer learning models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_transforms(img_size: int = 224):
    """
    Returns train and test torchvision transforms.
    
    Train transforms apply random flips and minor rotations to balance
    smaller defect representations and prevent overfitting.
    Test transforms apply deterministic resizing and normalization.
    """

    train_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ]
    )

    return train_transform, test_transform


def get_dataloaders(
    data_dir: str = "data",
    batch_size: int = 32,
    num_workers: int = 2,
    img_size: int = 224
):

    """
    Constructs train and test PyTorch DataLoaders from image folder directory structure.
    
    Expected directory tree:
      data/
        ├── train/
        │    ├── copper/
        │    ├── mousebite/
        │    └── ...
        └── test/
             ├── copper/
             └── ...
    """

    base_path = Path(data_dir)
    train_dir = base_path / "train"
    test_dir = base_path / "test"

    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"Expected 'train' and 'test' folders inside '{data_dir}'.\n"
            "Please verify that cropping pipeline ran successfully"
        )

    train_tf, test_tf = get_transforms(img_size = img_size)

    train_dataset = datasets.ImageFolder(root=str(train_dir), transform=train_tf)
    test_dataset = datasets.ImageFolder(root=str(test_dir), transform=test_tf)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, test_loader, train_dataset.classes


if __name__ == "__main__":
    print("Testing DataLoader pipeline...")
    train_loader, test_loader, classes = get_dataloaders(batch_size=16, num_workers=0)

    print(f"Detected Classes ({len(classes)}): {classes}")
    print(f"Train Batches: {len(train_loader)} | Test Batches: {len(test_loader)}")

    images, labels = next(iter(train_loader))
    print(f"Batch Image Tensor Shape: {images.shape} (Expected: [16, 3, 224, 224])")
    print(f"Batch Labels Shape: {labels.shape}")

