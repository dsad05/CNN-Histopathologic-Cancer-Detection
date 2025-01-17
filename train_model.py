import datetime
import json
import os
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet50_Weights, DenseNet121_Weights, EfficientNet_B0_Weights, Inception_V3_Weights

from preprocess_data import get_data_loaders


# Define model using updated way to load pretrained weights (ResNet18 for fast training)
print("Initializing model...")
def modify_last_layer(model, model_name):
    if model_name in ['resnet18', 'resnet50', 'inceptionv3']:
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif model_name == 'densenet121':
        model.classifier = nn.Linear(model.classifier.in_features, 2)
    elif model_name == 'efficientnetb0':
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    return model

def initialize_model_by_name_from_dictionary(model_name, model_dict):
    if model_name in model_dict:
        model = model_dict[model_name]()
        model = modify_last_layer(model, model_name) # Modify last layer
        print(f"{model_name} initialized.")
    else:
        raise ValueError(f"Model '{model_name}' not defined in the model dictionary.")
    return model


# Function to evaluate accuracy and loss
def train_validation_model(model, criterion, data_loader,device):
    print("Evaluating model...")
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    with torch.no_grad():  # Disable gradient calculation for evaluation
        for batch_idx, (images, labels) in enumerate(data_loader):
            images,labels = images.to(device), labels.to(device)
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
def train_and_evaluate(model, optimizer, criterion, train_loader, val_loader, epochs,device):
    print("Training loop started...")
    train_losses = []
    val_losses = []
    val_accuracies = []
    total_start_time = time.time()
    for epoch in range(epochs):
        epoch_start_time = time.time()
        print(f"Epoch {epoch + 1}/{epochs} started...")
        model.train()
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images,labels = images.to(device), labels.to(device)
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
        val_loss, val_accuracy = train_validation_model(model, criterion, val_loader,device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        # Print progress
        print(f"Epoch {epoch + 1} complete. Duration: {epoch_duration:.2f} seconds.")
        print(f"Training Loss: {train_losses[-1]}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}%")

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f"Training loop finished. Total duration: {total_duration:.2f} seconds.")
    return train_losses, val_losses, val_accuracies

def plot_train_metrics(train_losses, val_losses, val_accuracies, model_name, save_dir, current_time):
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