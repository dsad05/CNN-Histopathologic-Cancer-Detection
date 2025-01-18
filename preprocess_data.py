import os

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os


class CustomDataset(Dataset):
    def __init__(self, data_dir, labels_df, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # Printing the transform being used (with repr to show the object clearly)
        if self.transform:
            print(f"Used transform: {repr(self.transform)}")
        else:
            print("No transform is being used.")

        # List of image IDs and corresponding labels
        self.filenames = labels_df['id'].tolist()
        self.labels = labels_df['label'].tolist()

        # Create full paths for images
        self.full_filenames = [os.path.join(data_dir, f"{filename}.tif") for filename in
                               self.filenames]  # Assuming TIFF format

        # Default transform for converting PIL Image to tensor
        self.default_transform = transforms.Compose([
            transforms.ToTensor(),  # Convert image to tensor
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # Open image using PIL (this handles TIFF images)
        image = Image.open(self.full_filenames[idx]).convert("RGB")  # Convert to RGB to standardize the format

        if self.transform:
            image = self.transform(image)  # Apply custom transformations if provided
        else:
            image = self.default_transform(image)  # Apply default ToTensor transform if no custom transform is provided

        return image, self.labels[idx]

#
# train_transform = transforms.Compose([
#     transforms.RandomRotation(30),
#     transforms.RandomCrop(64),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip(),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2),
#     transforms.ToTensor(),  # Convert the image to a tensor
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the image
# ])

# val_transform = transforms.Compose([
#     transforms.ToTensor(),  # Convert the image to a tensor
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the image for validation
# ])

def get_data_loaders(data_dir, labels_file, total_size,  batch_size, train_ratio=0.7, val_ratio=0.15, use_existing_splits=False, splits_dir=None):
    if use_existing_splits:
        if splits_dir is None:
            raise ValueError("When using existing splits, splits_dir must be provided.")

        print("Using existing splits...")
        train_df = pd.read_csv(os.path.join(splits_dir, "train_split.csv"))
        val_df = pd.read_csv(os.path.join(splits_dir, "val_split.csv"))
        test_df = pd.read_csv(os.path.join(splits_dir, "test_split.csv"))
    else:
        labels_df = pd.read_csv(labels_file)

        # Ensure total_size does not exceed dataset size
        if total_size > len(labels_df):
            raise ValueError(f"Requested total_size={total_size} exceeds dataset size={len(labels_df)}")

        # Stratified sampling to select total_size samples
        _, sampled_df = train_test_split(
            labels_df, test_size=total_size, random_state=0, stratify=labels_df['label']
        )

        # Calculate sizes for train, validation, and test sets
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        test_size = total_size - train_size - val_size

        # Split the sampled data into train, validation, and test sets
        train_df, temp_df = train_test_split(
            sampled_df, test_size=(val_size + test_size), random_state=0, stratify=sampled_df['label']
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=test_size / (val_size + test_size), random_state=0, stratify=temp_df['label']
        )
        print("Saving train, val and test splits...")
        save_splits(train_df, val_df, test_df, splits_dir)

    print("Creating Train dataset...")
    train_dataset = CustomDataset(data_dir=data_dir, labels_df=train_df)

    print("Creating Validation dataset...")
    val_dataset = CustomDataset(data_dir=data_dir, labels_df=val_df)

    print("Creating Test dataset...")
    test_dataset = CustomDataset(data_dir=data_dir, labels_df=test_df)

    # Create DataLoaders for training, validation, and test sets
    print("Creating DataLoaders...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(
        f"Data split: {len(train_df)} training samples, {len(val_df)} validation samples, {len(test_df)} test samples.")

    return train_loader, val_loader, test_loader


def check_normalization(loader, loader_name):
    for images, labels in loader:
        print(f"--- {loader_name} ---")
        print(f"Shape: {images.shape}")
        print(f"Min: {images.min().item()}, Max: {images.max().item()}")
        print(f"Mean: {images.mean().item()}, Std: {images.std().item()}")

        # If data is in the range ~[-1, 1], assume it is normalized
        # (consistent with Normalize using mean=0.5 and std=0.5)
        if images.min() >= -1 and images.max() <= 1:
            print("Data is likely normalized.")
        else:
            print("Data is NOT normalized.")
        break  # Check only the first batch

def check_duplicates(train_loader, val_loader, test_loader):
    def get_all_ids(loader):
        """Extract all sample IDs from a DataLoader."""
        ids = []
        for images, labels in loader:
            # Assuming `labels` contains the unique IDs
            ids.extend(labels)  # Modify this if your IDs are in `labels` or another structure
        return set(ids)

    # Extract IDs from all DataLoaders
    train_ids = get_all_ids(train_loader)
    val_ids = get_all_ids(val_loader)
    test_ids = get_all_ids(test_loader)

    # Find duplicates between the sets
    train_val_duplicates = train_ids & val_ids
    train_test_duplicates = train_ids & test_ids
    val_test_duplicates = val_ids & test_ids

    # Print results
    if train_val_duplicates:
        print(f"Duplicates found between train and validation: {train_val_duplicates}")
    else:
        print("No duplicates between train and validation.")

    if train_test_duplicates:
        print(f"Duplicates found between train and test: {train_test_duplicates}")
    else:
        print("No duplicates between train and test.")

    if val_test_duplicates:
        print(f"Duplicates found between validation and test: {val_test_duplicates}")
    else:
        print("No duplicates between validation and test.")


def save_splits(train_df, val_df, test_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train_split.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val_split.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_split.csv"), index=False)
    print(f"Splits saved to {output_dir}")


def load_splits(split_dir):
    train_df = pd.read_csv(os.path.join(split_dir, "train_split.csv"))
    val_df = pd.read_csv(os.path.join(split_dir, "val_split.csv"))
    test_df = pd.read_csv(os.path.join(split_dir, "test_split.csv"))
    print("Splits loaded successfully.")
    return train_df, val_df, test_df
