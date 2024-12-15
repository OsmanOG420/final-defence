# Import necessary libraries
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Class labels
CLASS_LABELS = ['Agaricus', 'Blue_Oyster_Mushroom', 'Oyster_Mushroom', 
                'Phoenix_Oyster_Mushrooms', 'Phoenix_Oyster_Mushrooms', 
                'poisonous_mushroom']

# Load TFLite model
def load_model(tflite_model_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    return interpreter

# Preprocess the input image
def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image = np.array(image) / 255.0  # Normalize to range [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image.astype(np.float32)  # Ensure data type is float32
    return image

# Make prediction using the TFLite model
def predict(interpreter, image):
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']
    interpreter.set_tensor(input_index, image)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_index)
    return predictions

# Main app function
def main():
    st.title("Mushroom Image Classification")
    st.write("Upload an image of a mushroom, and the model will classify it.")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
    # Display uploaded image
       image = Image.open(uploaded_file)
       st.image(image, caption="Uploaded Image", use_container_width=True)
       st.write("Analyzing the image...")

       # Preprocess the image
       preprocessed_image = preprocess_image(image)

       # Load TFLite model
       interpreter = load_model("model.tflite")

       # Make prediction
       predictions = predict(interpreter, preprocessed_image)
       predicted_class = np.argmax(predictions)
       confidence = predictions[0][predicted_class]

       # Display results
       st.write(f"**Predicted Class:** {CLASS_LABELS[predicted_class]}")
       st.write(f"**Confidence:** {confidence:.2f}")
       st.bar_chart(predictions[0])


if __name__ == "__main__":
    main()
