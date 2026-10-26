# Plant Disease Prediction with CNN

A deep-learning image-classification project using Convolutional Neural Networks (CNNs), TensorFlow/Keras, and the PlantVillage leaf-image dataset to explore plant-disease classification.

---

## Project Overview

Plant diseases can severely impact crop yield and food security 🌾.  
This project aims to automatically identify plant diseases by analyzing images of plant leaves using a CNN-based image classification approach.

The model learns visual patterns such as:
- 🎨 Leaf color variations  
- 🔴 Spots, lesions, and discoloration  
- 🧬 Texture and shape features  

---

## Model Approach

- 📷 Image classification using **CNN**
- 🌿 Supports multiple plant species and disease categories
- 🔄 Uses **ImageDataGenerator** for preprocessing and data augmentation
- Validation performance should be reported from a reproducible held-out evaluation; no numeric metric is claimed here.

---

## Dataset

**PlantVillage Dataset (Kaggle)**  
🔗 https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset

### Dataset Details
- ✅ Healthy and diseased leaf images
- 🌱 Multiple plant species (Potato, Tomato, Apple, Corn, etc.)
- 🖼️ RGB images organized into class-wise folders

> ⚠️ **Note:**  
> The dataset is **not included** in this repository due to size limitations.  
> Please download it separately from Kaggle.

---

## Technologies

- 🐍 Python  
- 🧠 TensorFlow / Keras  
- 📊 NumPy  
- 📉 Matplotlib  
- 📸 OpenCV  
- 📓 Jupyter Notebook  

---

## Project Structure

```
Plant-Disease-Prediction-with-CNN/
│
├── Plant Disease Prediction with CNN.ipynb
├── README.md
├── .gitignore
└── requirements.txt
```


---

## Setup and Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/avviiiral/Plant-Disease-Prediction-with-CNN.git
cd Plant-Disease-Prediction-with-CNN
```
### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Download the Dataset

Download from Kaggle:
```
https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
```
Extract and organize the dataset as:
```
dataset/
├── train/
├── val/
└── test/

(Update dataset paths in the notebook if required.)
```

🚀 How to Run

1. Launch Jupyter Notebook:
 ```
jupyter notebook
```
2. Open the notebook:
```
Plant Disease Prediction with CNN.ipynb
```
3. Run all cells to:

📥 Load and preprocess the dataset

🧠 Train the CNN model

📊 Evaluate model performance

---

## Results and Evaluation

Evaluate the trained model on a held-out test set and report accuracy, per-class precision/recall, a confusion matrix, and any class imbalance. Results depend on the dataset split, preprocessing, and training settings. Do not interpret notebook results as field-ready agricultural diagnosis without external validation.

---

## Potential Improvements

🌐 Deploy the model using Flask or FastAPI

📱 Build a real-time prediction system (webcam or mobile app)

🧠 Apply Transfer Learning (ResNet, EfficientNet, MobileNet)

⚖️ Improve dataset balance and augmentation techniques

---

## License

This project is intended for educational and research purposes only.

---

## Author

Aviral Goyal
🔗 GitHub: https://github.com/avviiiral


## Project Keywords

`deep-learning` `cnn` `plant-disease-classification` `computer-vision` `tensorflow` `keras` `image-classification` `python`
