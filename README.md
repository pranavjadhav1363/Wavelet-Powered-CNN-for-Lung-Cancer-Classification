
# Wavelet-Powered-CNN-for-Lung-Cancer-Classification

Lung Cancer Classification with CNN and Wavelet Transforms
This repository contains the code and resources for a project on classifying lung cancer images (benign, malignant, and normal) using Convolutional Neural Networks (CNNs) enhanced with Wavelet Transform and Gaussian filtering.

# Project Structure
```
└── 📁Wavelet-Powered-CNN-for-Lung-Cancer-Classification
    └── 📁Bengin cases
    └── 📁DATA_PROCESSED
        └── 📁256x256_Guas
            └── 📁Benign
            └── 📁Malignant
            ├── 📁Normal
        └── 📁64x64
            └── 📁Benign
            └── 📁Malignant
            ├── 📁Normal
        └── 📁filters
            └── 📁Bengin                
            └── 📁Malignant
            └── 📁Normal
        └── 📁Pywavelet
            └── 📁Bengin
            └── 📁Malignant
            └── 📁Normal
            └── filters.txt
        └── 📁SVD
            └── 📁Bengin
            └── 📁Malignant
            ├── Normal
    └── 📁main
        └── 📁Model
            └── 📁firstmodel
                └── model.ipynb
                └── training1.xlsx
        └── DataPreProcessing_PART_1.ipynb
        └── DataPreProcessing_PART_2.ipynb
        └── DataPreProcessing_PART_3.ipynb
        └── DataPreProcessing_PART_4.ipynb
    └── 📁Malignant cases
    └── 📁Normal cases
    └── .gitignore
    └── info.txt
    └── README.md
```

# Project Overview

This project leverages image data from lung CT scans to classify three categories of lung cancer :

Benign
Malignant
Normal
The classification is carried out using a Convolutional Neural Network (CNN) with added pre-processing steps like Gaussian filtering and Wavelet Transforms to enhance image features. The final trained model outputs key metrics like accuracy, precision, recall, and ROC-AUC, and provides a confusion matrix for evaluation.

CHECK info.txt IN THE ROOT FOLDER FOR THE DATASET
# Setup Instructions
# # Requirements
The following Python packages are required to run the project:

numpy
pandas
opencv-python (cv2)
scikit-learn
tensorflow
pywavelets
matplotlib
You can install the dependencies using the following command:

bash
Copy code
pip install -r requirements.txt
Dataset
Place the lung CT scan images in the dataset/ folder, organized into subfolders as follows:

Benign/: Contains images classified as benign
Malignant/: Contains images classified as malignant
Normal/: Contains images classified as normal
Running the Project


Preprocessing: Run the scripts in the data_preprocessing/ folder sequentially to load, augment, and apply wavelet transformations and filters to the dataset:

PART_1: Loads and prepares the raw image data.
PART_2: Performs data augmentation to increase dataset size.
PART_3: Applies wavelet transformations to extract features.
PART_4: Enhances images using various filters (e.g., Gaussian).
Model Training: After preprocessing, run the script in the model/ folder to define and train the CNN model:

model.ipynb: Defines the CNN architecture and trains the model on the processed dataset. It also generates performance metrics and stores results in an Excel file.



# Model Summary
The CNN architecture includes:


This CNN architecture processes 64x64 grayscale images and is designed for multi-class classification with 3 output classes. Here's a breakdown of the architecture:

Input Layer:

The input shape is (64, 64, 1), indicating 64x64 pixels and 1 color channel (grayscale).
First Convolutional Block:

Conv2D(32, (3, 3)): Applies 32 filters of size 3x3 to the input image, followed by ReLU activation.
MaxPooling2D((2, 2)): Reduces the spatial dimensions by half (from 64x64 to 32x32).
Second Convolutional Block:

Conv2D(64, (3, 3)): Applies 64 filters of size 3x3, followed by ReLU activation.
MaxPooling2D((2, 2)): Further reduces the spatial dimensions (from 32x32 to 16x16).
Third Convolutional Block:

Conv2D(128, (3, 3)): Applies 128 filters of size 3x3, followed by ReLU activation.
MaxPooling2D((2, 2)): Again reduces the dimensions (from 16x16 to 8x8).
Flattening Layer:

Flatten(): Converts the 3D feature map (8x8x128) into a 1D vector (8192 units) for input to the dense layers.
Fully Connected (Dense) Layers:

Dense(512, activation='relu'): 512 neurons with ReLU activation.
Dense(256, activation='relu'): 256 neurons with ReLU activation.
Dense(128, activation='relu'): 128 neurons with ReLU activation.
Output Layer:

Dense(num_classes, activation='softmax'): Outputs probabilities for each of the 3 classes using the softmax function.
The model is trained using categorical_crossentropy loss and adam optimizer, with accuracy as the main evaluation metric.

Evaluation
The following metrics are calculated for model performance evaluation:

Accuracy
Precision
Recall
ROC-AUC
Confusion Matrix
Specificity
