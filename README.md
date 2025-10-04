# Handwritten Alphabet Prediction System

## Summary
A deep learning solution for recognizing handwritten English alphabets (A-Z) using Convolutional Neural Networks (CNN). This system combines a high-accuracy machine learning model (99.4% validation accuracy) with an intuitive GUI interface, enabling real-time predictions through digital drawing input. The project implements full ML pipeline management from data preprocessing to model deployment.

## Key Features
### Core Capabilities
- **Multi-Layer CNN Architecture**: 3x Conv2D + MaxPooling with Batch Normalization
- **Interactive GUI**: Natural drawing canvas with prediction visualization
- **Advanced Preprocessing**: Automatic normalization & one-hot encoding
- **Model Persistence**: Saved model weights for immediate inference
- **Performance Metrics**: Detailed accuracy/loss tracking

### Technical Highlights
- **Input Specifications**: 28x28 grayscale images
- **Output Classes**: 26 alphabets (A-Z)
- **Inference Speed**: <50ms per prediction
- **Validation Strategy**: Stratified 80-20 split
- **Augmentations**: Built-in dropout layers for regularization

## Installation Guide

### Prerequisites
- Python 3.8+ 
- Kaggle API credentials ([Setup Guide](https://github.com/Kaggle/kaggle-api))

```bash
# Clone repository
git clone https://github.com/bb30fps/handwritten-alphabet-prediction.git
cd handwritten-alphabet-prediction

# Install dependencies
pip install -r requirements.txt

# Download dataset (requires Kaggle API setup)
kaggle datasets download -d sachinpatel21/az-handwritten-alphabets-in-csv-format
unzip az-handwritten-alphabets-in-csv-format.zip -d data/raw/

Model Training
Data Pipeline
# 1. Preprocess data
python main/data_processing.py

# 2. Train model (automatically saves best weights)
python main/train.py
Training Parameters

Parameter	Value
Epochs - 100
Batch Size - 256
Learning Rate - 0.0001
Validation Split - 20%
Early Stopping - Patience=10

GUI Application
Launch the drawing interface:

bash
python main/gui.py
GUI Features

400x400 pixel drawing canvas

Real-time stroke visualization

Prediction confidence display

Auto-saving of drawings with predictions

Session-based output organization

GUI Interface Replace with actual screenshot

Project Structure
handwritten-alphabet-prediction/
├── data/                   : Data management
│   ├── raw/                : Original CSV files
│   └── processed/          : Preprocessed NPZ files
├── models/                 : Model storage
│   ├── best_model.keras    : Optimal weights
│   └── final_model.keras   : Final training result
├── main/                   : Source code
│   ├── data_processing.py  : Data pipeline
│   ├── model.py            : CNN architecture
│   ├── train.py            : Training script
│   ├── predict.py          : Inference module
│   └── gui.py              : Tkinter interface
├── outputs/                : Generated content
│   ├── plots/              : Training visualizations
│   └── drawings/           : User drawings+results
├── requirements.txt        : Dependency list
└── README.md               : Project guide

Contributing

We welcome contributions. Please follow these guidelines:
-Open an issue to discuss proposed changes
-Fork the repository
-Create feature branch (git checkout -b feature/NewFeature)
-Commit changes (git commit -m 'Add NewFeature')
-Push branch (git push origin feature/NewFeature)
-Submit Pull Request
-Development Standards
-PEP8 compliance
-Comprehensive docstrings
-Unit test coverage for new features
-Type hinting enforcement

Acknowledgements

Dataset: AZ Handwritten Alphabets by Sachin Patel

Core Libraries: TensorFlow, Keras, OpenCV, Pillow

GUI Framework: Tkinter

Visualization: Matplotlib, Seaborn

Contact
Project Maintainer
BAKI BORO
Email - geekbaki@gmail.com
