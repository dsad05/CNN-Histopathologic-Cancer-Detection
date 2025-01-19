from preprocess_data import *
from test_model import *
from train_model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epochs = 8
dataset_size = 220000
batch = 64*8

current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

model_name = 'inceptionv3'
model_dir = './models'
data_dir = './data/train'
labels_file = './data/train_labels.csv'
splits_dir = './data/splits'

save_dir = os.path.join("models", current_time)
os.makedirs(save_dir, exist_ok=True)

model_dict = {
    'resnet18': lambda: models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1),
    'resnet50': lambda: models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1),
    'densenet121': lambda: models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1),
    'efficientnetb0': lambda: models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1),
    'mobilenetv2': lambda: models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1),
    'efficientnetv2_s': lambda: models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1),
    'efficientnetv2_m': lambda: models.efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1),
    'convnext_tiny': lambda: models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
}

print("Starting program...")
print("Loading data...")
train_loader, val_loader, test_loader = get_data_loaders(data_dir, labels_file, dataset_size, batch, use_existing_splits=True, splits_dir=splits_dir)
print("Data loaded successfully.")

# print("Checking normalization...")
# check_normalization(train_loader, "Train Loader")
# check_normalization(val_loader, "Validation Loader")
# check_normalization(test_loader, "Test Loader")
#
# check_duplicates(train_loader, val_loader, test_loader)

#TRAINING
print("TRAINING PHASE STARTED")
print(f"Model initialization {model_name}...")
model = initialize_model_by_name_from_dictionary(model_name, model_dict).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_start_time = time.time()

print("Starting training...")
train_losses, val_losses, val_accuracies = train_and_evaluate(model, optimizer, criterion, train_loader, val_loader, epochs=epochs,device=device)

train_end_time = time.time()
train_duration = train_end_time - train_start_time

print("Training complete.")
# Save model with date and time in the filename inside the created directory
print("Saving model...")
model_filename = os.path.join(save_dir, f"{model_name}_cancer_{current_time}.pth")
torch.save(model.state_dict(), model_filename)
print(f"Model saved to {model_filename}.")
plot_train_metrics(train_losses, val_losses, val_accuracies, model_name, save_dir, current_time)


#TESTING
print("TESTING PHASE STARTED")
print("Loading model...")
print(f"Using the latest model directory: {model_filename}")
model = load_weights_to_test_model(model, model_filename).to(device)
print(f"Loaded model: {model_name}")

# Test the model
print("Starting model evaluation...")

test_start_time = time.time()

test_avg_loss, test_accuracy, correct_samples, misclassified_samples = test_model(model, test_loader, criterion, device=device)

test_end_time = time.time()
test_duration = test_end_time - test_start_time

print(f"Test set accuracy: {test_accuracy:.2f}%")

# Visualize correctly and incorrectly classified samples
visualize_samples(correct_samples, "Correctly Classified Samples", save_dir, model_name, current_time)
visualize_samples(misclassified_samples, "Misclassified Samples", save_dir, model_name, current_time)

# # Plot the results
plot_test_results(test_avg_loss, test_accuracy, correct_samples, misclassified_samples, model_name, save_dir, current_time)

save_model_configuration(model_name, save_dir, current_time, dataset_size, batch, train_losses, val_losses, val_accuracies, model, optimizer, criterion, train_loader, val_loader, epochs, test_accuracy, test_avg_loss, train_duration, test_duration)
