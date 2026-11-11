
# Pneumonia Detection from Chest X-rays

A deep learning project that uses Convolutional Neural Networks (CNNs) to classify chest X-ray images into two categories: **NORMAL** and **PNEUMONIA**.

## Project Overview

Pneumonia is a lung infection that can be identified through chest X-ray imaging. This project explores how deep learning and computer vision can help classify chest X-ray images automatically.

The model is built using TensorFlow and Keras and trained on a publicly available chest X-ray dataset.

## Objectives

- Build a CNN-based image classification model.
- Preprocess and normalize chest X-ray images.
- Train the model to distinguish NORMAL and PNEUMONIA images.
- Evaluate model performance using standard classification metrics.
- Visualize predictions and analyze classification errors.

## Technology Stack

- **Programming Language:** Python
- **Deep Learning:** TensorFlow, Keras
- **Data Processing:** NumPy
- **Visualization:** Matplotlib
- **Model Evaluation:** Scikit-learn
- **Development Environment:** VS Code
- **Version Control:** Git and GitHub

## Dataset

The project uses the Chest X-Ray Images (Pneumonia) dataset.

Dataset structure:

```text
data/
└── raw/
    └── chest_xray/
        ├── train/
        │   ├── NORMAL/
        │   └── PNEUMONIA/
        ├── test/
        │   ├── NORMAL/
        │   └── PNEUMONIA/
        └── val/
            ├── NORMAL/
            └── PNEUMONIA/
```

Dataset distribution:

| Dataset | NORMAL | PNEUMONIA |
|---|---:|---:|
| Training | 1,341 | 3,875 |
| Testing | 234 | 390 |
| Validation | 8 | 8 |

Total images: 5,856

## Model Architecture

A Convolutional Neural Network (CNN) was developed using TensorFlow/Keras.

Main components:

1. Convolutional layers for extracting image features.
2. MaxPooling layers for reducing spatial dimensions.
3. Flatten layer for converting feature maps into a vector.
4. Dense layer for classification learning.
5. Dropout layer to help reduce overfitting.
6. Sigmoid output layer for binary classification.

Input image size: 150 × 150 × 3

Output classes:
- NORMAL
- PNEUMONIA

## Model Performance

The original CNN model achieved the following results on the test dataset:

| Metric | Score |
|---|---:|
| Accuracy | 79.33% |
| Pneumonia Precision | 75.74% |
| Pneumonia Recall | 98.46% |
| Pneumonia F1-score | 85.62% |

### Confusion Matrix

| Actual / Predicted | NORMAL | PNEUMONIA |
|---|---:|---:|
| NORMAL | 111 | 123 |
| PNEUMONIA | 6 | 384 |

### Performance Analysis

- The model correctly identified 384 pneumonia images.
- It correctly classified 111 normal images.
- It incorrectly classified 123 normal images as pneumonia.
- It incorrectly classified 6 pneumonia images as normal.

The model achieved high pneumonia recall but struggled to correctly identify some normal images.

## Project Structure

```text
pneumonia-detection/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── best_model.keras
│   ├── final_model.keras
│   ├── best_improved_model.keras
│   └── final_improved_model.keras
│
├── notebooks/
│
├── reports/
│   ├── evaluation_results.md
│   ├── model_comparison.md
│   ├── performance_analysis.md
│   ├── confusion_matrix.png
│   └── predictions_visualization.png
│
├── src/
│   ├── inspect_data.py
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   ├── train_improved.py
│   ├── evaluate.py
│   ├── evaluate_improved.py
│   ├── visualize_predictions.py
│   └── analyze_performance.py
│
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/vanshika-2105/pneumonia-detection.git
cd pneumonia-detection
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install tensorflow matplotlib scikit-learn
```

NumPy is installed as a TensorFlow dependency.

### 5. Add the dataset

Download the Chest X-Ray Images (Pneumonia) dataset and place the extracted `chest_xray` folder inside:

```text
data/raw/chest_xray/
```

### 6. Run the project

From the project root:

```bash
python .\src\train.py
```

Evaluate the trained model:

```bash
python .\src\evaluate.py
```

Visualize predictions:

```bash
python .\src\visualize_predictions.py
```

Analyze model performance:

```bash
python .\src\analyze_performance.py
```

## Limitations

- The validation dataset contains only 16 images, so validation metrics are not reliable.
- The model produces false-positive predictions for some normal X-rays.
- The model has not undergone clinical validation.
- Results are dependent on the dataset and training setup.

## Future Improvements

- Use a larger and more diverse dataset.
- Improve normal-image classification performance.
- Experiment with transfer learning architectures such as MobileNetV2 and ResNet50.
- Apply data augmentation and further model evaluation.
- Explore explainable AI techniques such as Grad-CAM.

## Disclaimer

This project is developed for educational and research purposes only. It is not a medical diagnostic tool and must not be used to make clinical decisions.

## Author

**Vanshika Prajapati**

GitHub: [vanshika-2105](https://github.com/vanshika-2105)
