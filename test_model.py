import glob
import os
import json
import torch
from matplotlib import pyplot as plt
from torchvision import models
from torchvision.transforms import ToPILImage

# Load weights from the saved model
def load_weights_to_test_model(model, latest_model_dir):
    model.load_state_dict(torch.load(latest_model_dir))
    model.eval()
    return model

# Function to test the model
def test_model(model, data_loader, criterion, device):
    correct_preds = 0
    total_preds = 0
    running_loss = 0.0
    misclassified_samples = []
    correct_samples = []

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)  # Calculate loss
            running_loss += loss.item()

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
    avg_loss = running_loss / len(data_loader)
    return avg_loss, accuracy, correct_samples, misclassified_samples

# Function to visualize samples
def visualize_samples(samples, title, save_dir, model_name, current_time, n=6):
    to_pil = ToPILImage()
    fig, axes = plt.subplots(1, n, figsize=(15, 5))
    for i, (image, label, prediction) in enumerate(samples[:n]):
        ax = axes[i]
        ax.imshow(to_pil(image))
        ax.axis('off')
        ax.set_title(f"True: {label}\nPred: {prediction}")
    plt.suptitle(title)
    plt.tight_layout()

    plot_filename = os.path.join(save_dir, f"{model_name}_{title}_cancer_{current_time}_plot.png")
    plt.savefig(plot_filename)
    print(f"Saved visualization to {save_dir}")
    plt.show()

# Function to plot test results
def plot_test_results(avg_loss, accuracy, correct_samples, misclassified_samples, model_name, save_dir, current_time):
    plt.figure(figsize=(8, 6))
    categories = ["Correct", "Misclassified"]
    values = [len(correct_samples), len(misclassified_samples)]

    plt.bar(categories, values, color=["green", "red"])
    plt.ylabel("Number of Samples")
    title = f"Test Results - Accuracy: {accuracy:.2f}% - Avg Loss: {avg_loss:.4f}"
    plt.title(title)

    plot_filename = os.path.join(save_dir, f"{model_name}_{title}_cancer_{current_time}_plot.png")
    plt.savefig(plot_filename)
    plt.show()

def save_model_configuration(model_name, save_dir, current_time, dataset_size, batch_size, train_losses, val_losses, val_accuracies, model, optimizer, criterion, train_loader, val_loader, epochs, accuracy, avg_loss, train_duration, test_duration):
    # Save model configuration
    config = {
        'model_name': model_name,
        'timestamp': current_time,
        'epochs': epochs,
        'dataset_size': dataset_size,
        'batch_size': batch_size,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'test_accuracy': accuracy,
        'test avg_loss': avg_loss,
        'train duration': train_duration,
        'test duration': test_duration
    }
    config_filename = os.path.join(save_dir, f"{model_name}_cancer_{current_time}_config.json")
    with open(config_filename, 'w') as config_file:
        json.dump(config, config_file)

    print(f"Configuration saved to {config_filename}")