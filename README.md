# Bird Species Classification

A deep learning project for classifying 200 bird species using transfer learning on the Caltech-UCSD Birds 200 (CUB-200) dataset.  
The project compares three pre-trained models - **EfficientNetB0**, **MobileNetV2**, and **ResNet50V2** - with various hyperparameters, augmentation, and regularization techniques.

## Dataset
- **Name**: [CUB-200](https://www.vision.caltech.edu/datasets/cub_200_2011/)
- **Classes**: 200 bird species
- **Size**: 4,829 training images, 1,204 testing images
- Images are annotated with class labels

## Features
- **Transfer learning** with EfficientNetB0, MobileNetV2, and ResNet50V2
- **Data preprocessing**: resizing to (224, 224), normalization
- **Data augmentation**: rotation, shifts, shearing, zoom, brightness adjustment, horizontal flips
- **Class balancing** with `RandomOverSampler`
- **Regularization**: L2 regularization, Dropout
- **Callbacks**: ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
- **Evaluation**: Top-1 Accuracy and Average Accuracy per Class

## Results Summary
| Model           | Version | Val Acc (%) | Test Top-1 Acc (%) | Test Avg/Class (%) |
|----------------|---------|-------------|--------------------|--------------------|
| EfficientNetB0 | v1      | 40.10       | 36.05              | 35.49              |
| MobileNetV2    | v2      | 37.89       | 36.63              | 36.53              |
| ResNet50V2     | v2      | 37.34       | 35.96              | 35.41              |

**Best model:** EfficientNetB0 (v1) — fastest convergence and best accuracy given limited resources.
