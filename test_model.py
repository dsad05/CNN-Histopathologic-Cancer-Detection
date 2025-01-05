import glob
import os

import matplotlib.pyplot as plt
import torch
from torchvision import models
from torchvision.models import ResNet18_Weights
from torchvision.transforms import ToPILImage

from preprocess_data import get_data_loaders  # Import the existing function

# Paths to data and model
data_dir = './data/train'
labels_file = './data/train_labels.csv'
models_dir = './models'

# Function to find the latest saved model
def get_latest_model(models_dir):
    model_files = glob.glob(os.path.join(models_dir, "resnet18_cancer_*.pth"))
    if not model_files:
        raise FileNotFoundError("No saved models found in the specified directory!")
    return max(model_files, key=os.path.getctime)

# Load the latest model
latest_model_path = get_latest_model(models_dir)
print(f"Loaded model: {latest_model_path}")

# Extract date and time from model filename for saving plots
model_filename = os.path.basename(latest_model_path)
model_datetime = model_filename.split("_")[-2] + "_" + model_filename.split("_")[-1].split(".")[0]
# Example: '2025-01-05_12-30-45'

# Load the model
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Adjust for binary classification
model.load_state_dict(torch.load(latest_model_path))
model.eval()  # Switch to evaluation mode

# Create DataLoader for the test dataset using the existing function
train_loader, val_loader, test_loader = get_data_loaders(data_dir, labels_file)

# Function to test the model
def test_model(model, data_loader):
    correct_preds = 0
    total_preds = 0
    misclassified_samples = []
    correct_samples = []

    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

            # Collect correctly and incorrectly classified samples
            for i in range(len(images)):
                if predicted[i] == labels[i]:
                    correct_samples.append((images[i], labels[i], predicted[i]))
                else:
                    misclassified_samples.append((images[i], labels[i], predicted[i]))

    accuracy = correct_preds / total_preds * 100
    return accuracy, correct_samples, misclassified_samples

# Test the model
print("Starting model evaluation...")
test_accuracy, correct_samples, misclassified_samples = test_model(model, test_loader)
print(f"Test set accuracy: {test_accuracy:.2f}%")

# Function to visualize samples
def visualize_samples(samples, title, n=6, save_path=None):
    to_pil = ToPILImage()
    fig, axes = plt.subplots(1, n, figsize=(15, 5))
    for i, (image, label, prediction) in enumerate(samples[:n]):
        ax = axes[i]
        ax.imshow(to_pil(image))
        ax.axis('off')
        ax.set_title(f"True: {label}\nPred: {prediction}")
    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
    plt.show()

# Function to plot test results
def plot_test_results(accuracy, correct_samples, misclassified_samples, save_path=None):
    plt.figure(figsize=(8, 6))
    categories = ["Correct", "Misclassified"]
    values = [len(correct_samples), len(misclassified_samples)]

    plt.bar(categories, values, color=["green", "red"])
    plt.ylabel("Number of Samples")
    plt.title(f"Test Results - Accuracy: {accuracy:.2f}%")

    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    plt.show()

# Directory to save the plots and visualizations
save_dir = os.path.join(models_dir, model_datetime)
os.makedirs(save_dir, exist_ok=True)

# Visualize correctly and incorrectly classified samples
visualize_samples(correct_samples, "Correctly Classified Samples", save_path=os.path.join(save_dir, f"correct_samples_{model_datetime}.png"))
visualize_samples(misclassified_samples, "Misclassified Samples", save_path=os.path.join(save_dir, f"misclassified_samples_{model_datetime}.png"))

# Plot the results
plot_test_results(test_accuracy, correct_samples, misclassified_samples, save_path=os.path.join(save_dir, f"test_results_{model_datetime}.png"))
