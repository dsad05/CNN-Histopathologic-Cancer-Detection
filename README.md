# Histopathologic Cancer Detection Project  

This project was developed as part of the **Multimedia Techniques** course at Warsaw University of Technology (WUT). It is inspired by the Kaggle competition **[Histopathologic Cancer Detection](https://www.kaggle.com/competitions/histopathologic-cancer-detection/data)**.  

The main objective is to classify histopathology images as **positive** (containing tumor tissue) or **negative** (no tumor tissue).  

## Dataset  

The project uses the dataset provided by Kaggle. It consists of pathology image patches labeled based on whether the **central 32x32px region** contains tumor tissue. Tumor tissue in the outer regions does not affect the labels.  

### Dataset Structure:  
- **train/**: Folder containing labeled training images.  
- **test/**: Folder containing unlabeled test images for prediction.  
- **train_labels.csv**: A CSV file mapping image IDs to their respective labels (0 for negative, 1 for positive).  

## How to Run  

### 1. Clone the repository:  
```bash  
git clone https://github.com/yourusername/histopathologic-cancer-detection.git  
```  

### 2. Download the dataset from [Kaggle](https://www.kaggle.com/competitions/histopathologic-cancer-detection/data) and place it in the `data/` folder.  

### 3. Run the main script:  
```bash  
python main.py  
```  

The script will automatically load the dataset, split it into training, validation, and test sets, train a model, and evaluate its performance. You can change the model architecture and training settings by modifying the variables in the script.

### 4. Expected outputs:  
- The script will save the trained model with the current timestamp in the `models/` directory.
- The script will visualize and save plots for training/validation losses and accuracies, as well as test set results (accuracy and misclassified samples).

---

## Main Script (`main.py`)

The `main.py` script is the core of this project. It manages the process of loading the dataset, training a model, evaluating its performance, and saving results. Below is a breakdown of the main steps involved:

### Workflow of `main.py`:

1. **Importing Required Modules:**
   - The script imports functions from the following files to handle different tasks:
     - `preprocess_data.py`: For loading and preprocessing the dataset.
     - `train_model.py`: For training the model.
     - `test_model.py`: For evaluating the model.

2. **Setting Up the Device:**
   - The script checks for the availability of a GPU and sets the device accordingly. If a GPU is available, it will use CUDA for faster training.

   ```python
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   ```

3. **Model Initialization:**
   - The script includes a dictionary, `model_dict`, that maps model names (like ResNet50, DenseNet121, etc.) to their respective model initialization functions. This allows easy switching between different pre-trained models.
   
   ```python
   model_dict = {
       'resnet18': lambda: models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1),
       'resnet50': lambda: models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1),
       'densenet121': lambda: models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1),
       'efficientnetb0': lambda: models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1),
       'inceptionv3': lambda: models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
   }
   ```

4. **Loading Data:**
   - The `get_data_loaders` function is called to load the dataset and split it into training, validation, and test sets. 
   - If the `use_existing_splits=True` flag is passed, it loads pre-saved splits from the `splits_dir`, ensuring the same data split is used across experiments. If set to `False`, the data is split from scratch and saved for future use.

   ```python
   train_loader, val_loader, test_loader = get_data_loaders(
       data_dir, labels_file, dataset_size, batch, use_existing_splits=True, splits_dir=splits_dir
   )
   ```

5. **Training Phase:**
   - The model is initialized based on the selected model architecture.
   - The script uses **CrossEntropyLoss** as the loss function and **Adam optimizer** for training.
   - The model is trained for the specified number of epochs, and both training and validation losses/accuracies are recorded.
   - After training, the model is saved to a file with a timestamp in the filename.

   ```python
   model = initialize_model_by_name_from_dictionary(model_name, model_dict).to(device)
   train_losses, val_losses, val_accuracies = train_and_evaluate(
       model, optimizer, criterion, train_loader, val_loader, epochs=epochs, device=device
   )
   torch.save(model.state_dict(), model_filename)
   ```

6. **Testing Phase:**
   - The trained model is loaded from the saved file, and its performance is evaluated on the test set.
   - The test set accuracy is reported, and both correctly and incorrectly classified samples are visualized.

   ```python
   model = load_weights_to_test_model(model, model_filename)
   test_avg_loss, test_accuracy, correct_samples, misclassified_samples = test_model(model, test_loader, criterion)
   ```

7. **Saving Results:**
   - After testing, the script saves detailed results, including test accuracy and loss, as well as visualizations of misclassified and correctly classified samples.
   - Model configurations are also saved for later reference, which includes hyperparameters, model architecture details, and the results of the training and testing phases.

   ```python
   plot_test_results(test_avg_loss, test_accuracy, correct_samples, misclassified_samples, model_name, save_dir, current_time)
   save_model_configuration(model_name, save_dir, current_time, dataset_size, batch, train_losses, val_losses, val_accuracies, model, optimizer, criterion, train_loader, val_loader, epochs, test_accuracy, test_avg_loss)
   ```

## Acknowledgements  

This project is inspired by the Kaggle competition **[Histopathologic Cancer Detection](https://www.kaggle.com/competitions/histopathologic-cancer-detection/data)** and utilizes its dataset. Special thanks to the PCam team for providing the benchmark data.