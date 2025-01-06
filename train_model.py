import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import ResNet18_Weights  # Import weights class for ResNet18

from preprocess_data import get_data_loaders

EPOCHS = 15

# Paths to data and labels
data_dir = './data/train'
labels_file = './data/train_labels.csv'

print("Starting program...")

# Create DataLoaders for train, val, and test datasets using the preprocess_data function
print("Loading data...")
train_loader, val_loader, test_loader = get_data_loaders(data_dir, labels_file)
print("Data loaded successfully.")

# Define model using updated way to load pretrained weights (ResNet18 for fast training)
print("Initializing model...")
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # Load pretrained weights for ResNet18
model.fc = nn.Linear(model.fc.in_features, 2)  # Modify for binary classification
print("Model initialized.")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Function to evaluate accuracy and loss
def evaluate_model(model, data_loader):
    print("Evaluating model...")
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    with torch.no_grad():  # Disable gradient calculation for evaluation
        for batch_idx, (images, labels) in enumerate(data_loader):
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

    avg_loss = running_loss / len(data_loader)
    accuracy = correct_preds / total_preds * 100
    print("Evaluation complete.")
    return avg_loss, accuracy

# Training loop
def train_and_evaluate(model, train_loader, val_loader, epochs):
    print("Training loop started...")
    train_losses = []
    val_losses = []
    val_accuracies = []
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs} started...")
        model.train()
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            #print(f"Training batch {batch_idx + 1}/{len(train_loader)}...")
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Save training loss
        train_losses.append(running_loss / len(train_loader))

        # Evaluate on validation set after each epoch
        print(f"Epoch {epoch + 1} validation started...")
        val_loss, val_accuracy = evaluate_model(model, val_loader)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Print progress
        print(f"Epoch {epoch + 1} complete. Training Loss: {train_losses[-1]}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}%")

    print("Training loop finished.")
    return train_losses, val_losses, val_accuracies

# Train the model
print("Starting training...")
train_losses, val_losses, val_accuracies = train_and_evaluate(model, train_loader, val_loader, EPOCHS)
print("Training complete.")

# Get current date and time
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Save model with date and time in the filename
print("Saving model...")
model_filename = f"models/resnet18_cancer_{current_time}.pth"
torch.save(model.state_dict(), model_filename)
print(f"Model saved to {model_filename}.")

# Plot loss and accuracy
print("Plotting results...")
plt.figure(figsize=(10, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, '-o', label="Training Loss")
plt.plot(val_losses, '-o', label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(val_accuracies, '-o', label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Validation Accuracy")
plt.legend()

plt.tight_layout()

plot_filename = f"models/resnet18_cancer_{current_time}_plot.png"
plt.savefig(plot_filename)
print(f"Plot saved to {plot_filename}")

plt.show()
print("Plotting complete. Plot saved.")


