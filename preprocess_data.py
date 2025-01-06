import os

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

dataset_size = 5000
BATCH = 64

# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, data_dir, labels_df, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # List of image IDs and corresponding labels
        self.filenames = labels_df['id'].tolist()
        self.labels = labels_df['label'].tolist()

        # Create full paths for images
        self.full_filenames = [os.path.join(data_dir, f"{filename}.tif") for filename in self.filenames]  # Assuming TIFF format

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # Open image using PIL (this handles TIFF images)
        image = Image.open(self.full_filenames[idx]).convert("RGB")  # Convert to RGB to standardize the format

        if self.transform:
            image = self.transform(image)  # Apply transformations if any
        return image, self.labels[idx]


# Define transformations (for tensor inputs)
train_transform = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomCrop(64),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the image
])

val_transform = transforms.Compose([
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the image for validation
])


def get_data_loaders(data_dir, labels_file, total_size=dataset_size, train_ratio=0.7, val_ratio=0.15, batch_size=BATCH):
    # Load the labels file
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

    # Create datasets for train, validation, and test
    train_dataset = CustomDataset(data_dir=data_dir, labels_df=train_df, transform=train_transform)
    val_dataset = CustomDataset(data_dir=data_dir, labels_df=val_df, transform=val_transform)
    test_dataset = CustomDataset(data_dir=data_dir, labels_df=test_df, transform=val_transform)

    # Create DataLoaders for training, validation, and test sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(
        f"Data split completed: {len(train_df)} training samples, {len(val_df)} validation samples, {len(test_df)} test samples.")

    return train_loader, val_loader, test_loader