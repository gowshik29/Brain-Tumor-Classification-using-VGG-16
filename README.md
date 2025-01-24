# Brain Tumor Detection using VGG16

## Overview

This project uses the VGG16 pre-trained model to classify brain tumor images into two categories: No (no tumor) and Yes (tumor detected). The model is fine-tuned and trained on a dataset of brain tumor MRI images. The dataset is split into training, validation, and testing sets to ensure robust model evaluation.

## Features

Preprocessing of input images, including resizing, cropping, and augmentation.

Data visualization for understanding class distribution and sample images.

Utilization of VGG16 as a feature extractor with custom classification layers.

Image augmentation to improve model robustness.

Training and validation performance tracking.

Early stopping to prevent overfitting.

## Prerequisites

Before running the code, ensure you have the following installed:

Python 3.7+

TensorFlow 2.x

Keras

OpenCV

NumPy

Matplotlib

Plotly

scikit-learn

tqdm

imutils

## Dataset

The dataset should have the following structure:

brain_tumor_dataset/
    NO/
        image1.jpg
        image2.jpg
        ...
    YES/
        image1.jpg
        image2.jpg
        ...

# Steps to Run the Code

### 1. Clone the Repository

git clone https://github.com/your_username/brain-tumor-detection-vgg16.git
cd brain-tumor-detection-vgg16

### 2. Install Dependencies

pip install -r requirements.txt

### 3. Prepare Dataset

Ensure the dataset is placed in the appropriate directory (e.g., brain_tumor_dataset/).

### 4. Split the Dataset

The script will automatically split the dataset into training, validation, and testing sets:

splitfolders.ratio(
    "brain_tumor_dataset/",
    output="output/",
    seed=42,
    ratio=(0.8, 0.1, 0.1)
)

### 5. Train the Model

Run the script to preprocess images, train the model, and save it:

python train_model.py

### 6. Visualize Results

The training script will generate:

Accuracy and loss plots

Visualizations of augmented images

Preprocessed sample images

### 7. Use the Model

Load the saved model for inference:

from tensorflow.keras.models import load_model

model = load_model('brain_tumor_vgg16_model.h5')
prediction = model.predict(test_image)

## Directory Structure

.
├── brain_tumor_dataset/   # Original dataset
├── output/                # Processed dataset splits
│   ├── train/
│   ├── val/
│   └── test/
├── preview/               # Augmented image previews
├── train_model.py         # Main training script
├── requirements.txt       # Dependencies
├── brain_tumor_vgg16_model.h5 # Saved model
└── README.md              # Project documentation

## Results

Training Accuracy: [Insert final training accuracy]

Validation Accuracy: [Insert final validation accuracy]

Test Accuracy: [Insert final test accuracy]

## Acknowledgments

This project uses the VGG16 model pre-trained on the ImageNet dataset. The dataset is sourced from [provide dataset source].

Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.
