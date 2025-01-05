# Histopathologic Cancer Detection Project  

This project was developed as part of the **Multimedia Techniques** course at Warsaw University of Technology (WUT). It is inspired by the Kaggle competition **[Histopathologic Cancer Detection](https://www.kaggle.com/competitions/histopathologic-cancer-detection/data)**.  

The main objective is to classify histopathology images as **positive** (containing tumor tissue) or **negative**.  

## Dataset  

The project uses the dataset provided by Kaggle. It consists of pathology image patches labeled based on whether the **central 32x32px region** contains tumor tissue. Tumor tissue in the outer regions does not affect the labels.  

### Dataset Structure:  
- **train/**: Folder containing labeled training images.  
- **test/**: Folder containing unlabeled test images for prediction.  
- **train_labels.csv**: A CSV file mapping image IDs to their respective labels (0 for negative, 1 for positive).  


## How to Run  

1. Clone the repository:  
```bash  
git clone https://github.com/yourusername/histopathologic-cancer-detection.git  
```  

2. Download the dataset from [Kaggle](https://www.kaggle.com/competitions/histopathologic-cancer-detection/data) and place it in the `data/` folder.  

3. Run the preprocessing script:  
```bash  
python preprocess_data.py  
```  

4. Train the model:  
```bash  
python train_model.py  
```  

5. Evaluate the model or make predictions:  
```bash  
python test_model.py  
```  

## Acknowledgements  

This project is inspired by the Kaggle competition **[Histopathologic Cancer Detection](https://www.kaggle.com/competitions/histopathologic-cancer-detection/data)** and utilizes its dataset. Special thanks to the PCam team for providing the benchmark data.  
