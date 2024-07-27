import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
model = tf.keras.models.load_model('cifar10_model.h5')
st.title("Image Classification App")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Classify"):
        st.write("Classifying...")
        # Preprocess the image
        image = image.resize((32, 32))
        image = np.array(image)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)

        # Make predictions
        predictions = model.predict(image)
        class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        class_index = np.argmax(predictions)
        st.write(f"Predicted Class: {class_names[class_index]}")
