# 🌱 Plant Disease Prediction with CNN

A deep learning project that uses **Convolutional Neural Networks (CNNs)** to detect and classify plant diseases from leaf images.  
This project is built using **TensorFlow / Keras** and trained on the **PlantVillage dataset**.

---

## 📌 Project Overview

Plant diseases can significantly affect crop yield and food security.  
This project aims to automatically identify plant diseases by analyzing images of plant leaves using a CNN-based image classification model.

The model learns visual patterns such as:
- Leaf color
- Spots and lesions
- Texture variations

---

## 🧠 Model Highlights

- Image classification using **CNN**
- Trained on multiple plant species and disease categories
- Uses **ImageDataGenerator** for data preprocessing and augmentation
- Achieves high validation accuracy on unseen data

---

## 📂 Dataset

**PlantVillage Dataset (Kaggle)**  
🔗 https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset

### Dataset Details
- Contains healthy and diseased leaf images
- Multiple plant species (e.g., Potato, Tomato, Apple, Corn, etc.)
- RGB images organized by class folders

> ⚠️ Dataset is **not included** in this repository due to size limitations.  
Please download it separately from Kaggle.

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- OpenCV
- Jupyter Notebook

---

## 📁 Project Structure

```
Plant-Disease-Prediction-with-CNN/
│
├── Plant Disease Prediction with CNN.ipynb
├── README.md
├── .gitignore
└── requirements.txt
```

---

## ⚙️ Setup & Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/avviiiral/Plant-Disease-Prediction-with-CNN.git
cd Plant-Disease-Prediction-with-CNN
```
2️⃣ Install dependencies
```
pip install -r requirements.txt
```
3️⃣ Download the dataset

Download from Kaggle:
```
https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
```
Extract and organize into folders like:
```
dataset/
  ├── train/
  ├── val/
  └── test/
```
(Adjust paths in the notebook if needed.)

🚀 How to Run

1. Open Jupyter Notebook:
```
jupyter notebook
```

2. Open:
```
Plant Disease Prediction with CNN.ipynb
```

3. Run all cells to:

  Load data

 Train the CNN model

  Evaluate performance

📊 Results

High training and validation accuracy

Low validation loss indicating good generalization

Model performs well on unseen plant leaf images

(Exact metrics may vary depending on hardware and training configuration.)

🔮 Future Improvements

Deploy model using Flask / FastAPI

Add real-time prediction via webcam or mobile app

Use Transfer Learning (ResNet, EfficientNet, MobileNet)

Improve dataset balance and augmentation

📜 License

This project is for educational and research purposes.

👨‍💻 Author

Aviral Goyal
🔗 GitHub: https://github.com/avviiiral
