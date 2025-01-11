import datetime
import json
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet50_Weights, DenseNet121_Weights, EfficientNet_B0_Weights, Inception_V3_Weights

from preprocess_data import get_data_loaders

EPOCHS = 5

model_name = 'densenet121'

# Dictionary of models to initialize
model_dict = {
    'resnet18': lambda: models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1),
    'resnet50': lambda: models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1),
    'densenet121': lambda: models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1),
    'efficientnetb0': lambda: models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1),
    'inceptionv3': lambda: models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
}

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
def modify_last_layer(model, model_name):
    if model_name in ['resnet18', 'resnet50', 'inceptionv3']:
        model.fc = nn.Linear(model.fc.in_features, 2)  # Dla ResNet, Inception
    elif model_name == 'densenet121':
        model.classifier = nn.Linear(model.classifier.in_features, 2)  # Dla DenseNet
    elif model_name == 'efficientnetb0':
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)  # Dla EfficientNet B0
    return model

if model_name in model_dict:
    model = model_dict[model_name]()
    model = modify_last_layer(model, model_name) # Modify last layer
    print(f"{model_name} initialized.")
else:
    raise ValueError(f"Model '{model_name}' not defined in the model dictionary.")

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

# Create a directory for saving the model and results
save_dir = os.path.join("models", current_time)
os.makedirs(save_dir, exist_ok=True)

# Save model with date and time in the filename inside the created directory
print("Saving model...")
model_filename = os.path.join(save_dir, f"{model_name}_cancer_{current_time}.pth")
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

plot_filename = os.path.join(save_dir, f"{model_name}_cancer_{current_time}_plot.png")
plt.savefig(plot_filename)
print(f"Plot saved to {plot_filename}")

plt.show()
print("Plotting complete. Plot saved.")

# Save model configuration
config = {
    'model_name': model_name,
    'timestamp': current_time
}
config_filename = os.path.join(save_dir, f"{model_name}_cancer_{current_time}_config.json")
with open(config_filename, 'w') as config_file:
    json.dump(config, config_file)

print(f"Configuration saved to {config_filename}")