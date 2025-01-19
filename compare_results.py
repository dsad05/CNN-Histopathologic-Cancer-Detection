import json
import matplotlib.pyplot as plt
import os

model_dir = './models'

def load_json_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def find_json_files_in_ok_dirs(model_dir):
    json_files = []
    for root, dirs, files in os.walk(model_dir):
        if "OK" in os.path.basename(root):
            for file in files:
                if file.endswith('.json'):
                    json_files.append(os.path.join(root, file))
    return json_files

def compare_models(model_json_files):
    model_names = []
    val_accuracies = []
    test_accuracies = []
    val_losses = []
    test_avg_losses = []
    train_durations = []
    epochs = []

    for file_path in model_json_files:
        model_data = load_json_data(file_path)
        model_names.append(model_data["model_name"])
        val_accuracies.append(model_data["val_accuracies"])
        test_accuracies.append(model_data["test_accuracy"])
        val_losses.append(model_data["val_losses"])
        test_avg_losses.append(model_data["test avg_loss"])
        train_durations.append(model_data["train duration"])
        epochs.append(model_data["epochs"])

    # Tworzenie wykresu porównawczego dokładności na zbiorze walidacyjnym
    plt.figure(figsize=(10, 6))
    for i, model_name in enumerate(model_names):
        plt.plot(range(epochs[i]), val_accuracies[i], label=f'{model_name} - Val Accuracy')

    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Porównanie dokładności walidacyjnej różnych modeli')
    plt.legend()
    plt.show()

    # Tworzenie wykresu porównawczego dokładności na zbiorze testowym
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, test_accuracies, color='skyblue')
    plt.xlabel('Model')
    plt.ylabel('Test Accuracy (%)')
    plt.ylim(90,100)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    bars = plt.bar(model_names, test_accuracies, color=['skyblue', 'orange', 'limegreen', 'salmon'])
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.2, f'{yval:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.title('Porównanie dokładności testowej różnych modeli')
    plt.show()

    # Tworzenie wykresu porównawczego strat testowych
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, test_avg_losses, color=['skyblue', 'orange', 'limegreen', 'salmon'])
    plt.xlabel('Model')
    plt.ylabel('Test Avg Loss')
    plt.title('Porównanie średnich strat testowych różnych modeli')
    plt.show()

    # Tworzenie wykresu porównawczego czasu treningu
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, train_durations, color=['skyblue', 'orange', 'limegreen', 'salmon'])
    plt.xlabel('Model')
    plt.ylabel('Train Duration (seconds)')
    plt.title('Porównanie czasu treningu różnych modeli')
    plt.show()

# Lista plików JSON z wynikami różnych modeli
model_json_files = [
    'path_to_model_1.json',  # Zmień na rzeczywiste ścieżki do plików JSON
    'path_to_model_2.json',
    'path_to_model_3.json'
]

model_json_files = find_json_files_in_ok_dirs(model_dir)
compare_models(model_json_files)
