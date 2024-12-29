# MNIST Fashion Classification Project

Welcome to the MNIST Fashion Classification Project! This repository contains the code and resources for training, evaluating, and deploying a Convolutional Neural Network (CNN) to classify images from the Fashion MNIST dataset. Additionally, a Gradio-based web application is included for real-time image classification.

---

## Overview

The MNIST Fashion dataset consists of 28x28 grayscale images of 10 categories of clothing and accessories. This project demonstrates:

1. Training a CNN to classify images into one of the 10 classes.
2. Visualizing training progress and model performance.
3. Deploying a Gradio-based web app for real-time classification.

Check out the live demo on Hugging Face Spaces: [Mnist-Fashion](https://huggingface.co/spaces/cisemh/Mnist-Fashion)

## Features

- **Training**: A CNN model is trained using TensorFlow and Keras.
- **Evaluation**: The trained model is evaluated on the test dataset.
- **Visualization**: Training accuracy, validation accuracy, and sample predictions are visualized.
- **Web App**: A Gradio interface for uploading images and receiving predictions.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/csm34/MNIST-Fashion.git
   cd MNIST-Fashion
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the trained model:
   Ensure `fashion_mnist_cnn_model.keras` is in the project directory. If you need to train the model yourself, refer to the [Model Training](#model-training) section.

---

## Usage

### Running the Gradio Web App

Launch the Gradio interface for real-time image classification:
```bash
python app.py
```

### Training the Model
To train the CNN from scratch, execute the `notebook.ipynb` file in a Jupyter environment. This will:
- Train the model on the Fashion MNIST dataset.
- Save the trained model as `fashion_mnist_cnn_model.keras`.

### Example Workflow
1. Train the model (if required).
2. Use `app.py` to classify new images via the Gradio web app.

---

## Project Structure

```plaintext
MNIST-Fashion/
├── app.py                # Gradio-based web app
├── fashionMnist.ipynb        # Jupyter notebook for model training and visualization
├── fashion_mnist_cnn_model.keras # Pre-trained model file
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
```

---

## Model Training

The CNN architecture includes:
- Convolutional layers with ReLU activation.
- Max pooling and Dropout layers to prevent overfitting.
- A fully connected layer with softmax activation for classification.

### Steps
1. Load and preprocess the Fashion MNIST dataset.
2. Build the CNN model using TensorFlow/Keras.
3. Train the model with `adam` optimizer and monitor accuracy.
4. Save the model for later use in the web app.

### Training Command
Run the Jupyter notebook to train and evaluate the model.

---

## Gradio Application

The Gradio web app allows users to:
- Upload a clothing image.
- Receive predictions with confidence scores for the top 3 classes.

### Key Components
- **Input**: Upload an image.
- **Output**: Predicted class, confidence score, and top 3 probabilities.

---

## Dataset

The Fashion MNIST dataset consists of:
- **Training Set**: 60,000 images
- **Test Set**: 10,000 images

### Classes

1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

---

## Acknowledgements

- Dataset: [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) by Zalando Research.
- TensorFlow and Keras for deep learning frameworks.
- Gradio for creating an interactive web interface.
