# 🌿 Plant Disease Classifier with CNN

A deep learning-based application to detect plant diseases using images. The system is built using PyTorch for training a Convolutional Neural Network (CNN) model and a custom-built GUI app using Tkinter for real-time prediction and diagnosis tracking.

---

## 📦 Project Overview

This repository contains two main components:

1. **Model Training & Evaluation Script**:

   - Trains a simple CNN model on a dataset of plant leaf images.
   - Evaluates performance using accuracy, precision, recall, F1-score, confusion matrix, ROC, and Precision-Recall curves.
   - Saves the trained model for later use.
2. **GUI Application**:

   - A user-friendly desktop application for uploading or capturing plant leaf images.
   - Predicts plant health/disease class in real time.
   - Displays treatment suggestions based on predictions.
   - Stores diagnosis history in an SQLite database.

---

## 🔍 Features

- ✅ End-to-end image classification pipeline
- 📊 Detailed evaluation metrics including ROC and PR curves
- 💾 Model saving/loading functionality
- 🖥️ Interactive GUI for image upload, live camera support, and result visualization
- 🗂️ Diagnosis history tracking with SQLite
- 🌱 Treatment suggestion knowledge base

---

## 🧠 Model Architecture

### CNN Structure

```python
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
```

### Input Size: `(3 x 128 x 128)`

### Optimizer: `Adam`

### Loss Function: `CrossEntropyLoss`

### Learning Rate Scheduler: `StepLR(gamma=0.1, step_size=7)`

---

## 📁 Dataset Requirements

Place your dataset in the following directory structure:

```
dataset/
├── train/
│   ├── TomatoEarlyBlight/
│   ├── TomatoHealthy/
│   └── ...
├── valid/
│   ├── TomatoEarlyBlight/
│   ├── TomatoHealthy/
│   └── ...
└── test/
    ├── TomatoHealthy3.jpg
    └── ...
```

---

## 📈 Performance Metrics

The trained model outputs the following metrics after validation:

- Accuracy
- Precision (Weighted)
- Recall (Weighted)
- F1 Score (Weighted)
- Classification Report
- Confusion Matrix
- Precision-Recall Curve per Class
- ROC Curve per Class

---

## 🖼️ GUI Application Features

### 📷 Upload Image

- Browse and select any local image file (.jpg, .jpeg, .png)

### 📸 Live Camera

- Capture and analyze leaf images directly from webcam

### 🕒 History

- View previous diagnoses with timestamps, predictions, and confidence scores

### 🩺 Treatment Suggestions

- Custom recommendations for each disease class:
  - Organic and chemical options
  - "No treatment needed" for healthy plants

---

## 🛠️ Dependencies

Install required packages using:

```bash
pip install torch torchvision torchaudio scikit-learn matplotlib seaborn pillow tqdm torchsummary opencv-python sqlite3
```

> Note: For CUDA support, use the appropriate index URL:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

---

## 🚀 How to Run

### 1. Train the Model

Run the training script from Jupyter Notebook or Python environment:

```bash
python train_model.py
```

Ensure `train/`, `valid/` folders exist with labeled data.

### 2. Launch the GUI App

Once the model is saved as `complete_model.pth`, run:

```bash
python app2.1.py
```

Or simply open and execute all cells in the notebook version.

---

## 📁 Saved Model Format

The model is saved with metadata:

```python
{
    'model_state_dict': model.state_dict(),
    'class_names': train_dataset.classes,
    'input_size': (3, 128, 128)
}
```

Make sure to redefine the exact architecture when loading.

---


## 📬 Contact

For questions, issues, or collaboration opportunities, please contact us at:

📧 muzankibu977.email@example.com
🌐 GitHub: https://github.com/muzankibu977

---

## 🙏 Acknowledgments

- PyTorch Team
- Scikit-learn Contributors
- OpenCV Community
- Tkinter Developers

---

Feel free to fork, contribute, or extend this project!😊
