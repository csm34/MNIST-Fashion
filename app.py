import gradio as gr
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('fashion_mnist_cnn_model.keras')

# Class names for the Fashion MNIST dataset
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

def predict_image(image):
    """Function to predict the class of a given image."""
    # Resize and normalize the input image
    image = image.resize((28, 28)).convert('L')
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)

    # Make predictions
    predictions = model.predict(image_array)
    predicted_label = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    return {class_names[i]: float(predictions[0][i]) for i in range(10)}, class_names[predicted_label], confidence

# Create a Gradio interface
demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", image_mode="L", label="Upload a clothing image"),
    outputs=[
        gr.Label(num_top_classes=3, label="Predictions"),  # Top 3 predictions with probabilities
        gr.Text(label="Predicted Label"),
        gr.Text(label="Confidence"),
    ],
    title="Fashion MNIST Classifier",
    description="Upload an image of a clothing item, and the model will predict the category."
)

# Launch the Gradio app
demo.launch()
