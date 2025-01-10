import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm


class AbstractArtDataset(Dataset):
    def __init__(self, root_dir, image_size=256, train=True):
        self.root_dir = Path(root_dir)
        self.train = train

        # Define transformations
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                (
                    transforms.RandomHorizontalFlip()
                    if train
                    else transforms.Lambda(lambda x: x)
                ),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        # Get all image files
        self.image_files = []
        valid_extensions = {".jpg", ".jpeg", ".png"}

        for ext in valid_extensions:
            self.image_files.extend(list(self.root_dir.glob(f"*{ext}")))

        print(f"Found {len(self.image_files)} images in {root_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]

        try:
            # Open image and convert to RGB
            image = Image.open(img_path).convert("RGB")

            # Apply transformations
            image = self.transform(image)

            return image

        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a random valid image instead
            return self[np.random.randint(len(self))]


def prepare_dataset(download_path, output_path, train_split=0.9, image_size=256):
    """
    Prepare the dataset by organizing and splitting the data
    """
    output_path = Path(output_path)

    # Create output directories
    train_dir = output_path / "train"
    val_dir = output_path / "val"

    for dir in [train_dir, val_dir]:
        dir.mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_files = []
    for ext in [".jpg", ".jpeg", ".png"]:
        image_files.extend(Path(download_path).glob(f"*{ext}"))

    # Shuffle files
    np.random.shuffle(image_files)

    # Split into train and validation
    split_idx = int(len(image_files) * train_split)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    # Copy files to respective directories
    def copy_files(files, dest_dir):
        for file in tqdm(files, desc=f"Copying to {dest_dir.name}"):
            shutil.copy2(file, dest_dir / file.name)

    copy_files(train_files, train_dir)
    copy_files(val_files, val_dir)

    return len(train_files), len(val_files)


def get_dataloaders(data_dir, batch_size=32, image_size=256, num_workers=4):
    """
    Create training and validation dataloaders
    """
    train_dataset = AbstractArtDataset(
        os.path.join(data_dir, "train"), image_size=image_size, train=True
    )

    val_dataset = AbstractArtDataset(
        os.path.join(data_dir, "val"), image_size=image_size, train=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def verify_dataset(dataloader):
    """
    Verify the dataset by checking a batch of images
    """
    batch = next(iter(dataloader))
    print(f"Batch shape: {batch.shape}")
    print(f"Value range: [{batch.min():.2f}, {batch.max():.2f}]")
    print(f"Mean: {batch.mean():.2f}")
    print(f"Std: {batch.std():.2f}")


if __name__ == "__main__":
    DOWNLOAD_PATH = "abstract_art"
    OUTPUT_PATH = "processed_abstract_art"

    n_train, n_val = prepare_dataset(DOWNLOAD_PATH, OUTPUT_PATH)
    train_loader, val_loader = get_dataloaders(OUTPUT_PATH)
    verify_dataset(train_loader)
