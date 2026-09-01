import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

st.set_page_config(page_title="Leopard vs Tiger Classifier", page_icon="🐅", layout="centered")

@st.cache_resource
def load_model():
    from tensorflow.keras.layers import InputLayer
    class SafeInputLayer(InputLayer):
        def __init__(self, *args, **kwargs):
            kwargs.pop('batch_shape', None)
            kwargs.pop('optional', None)
            super().__init__(*args, **kwargs)

    return tf.keras.models.load_model(
        'leopard_tiger_model.h5', 
        compile=False, 
        custom_objects={'InputLayer': SafeInputLayer}
    )
model = load_model()

st.title("🐅 Leopard vs Tiger Classifier 🐆")
st.write("Upload an image of a leopard or a tiger to test the model!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_display = Image.open(uploaded_file)
    st.image(image_display, caption='Uploaded Image', use_column_width=True)

    img = image_display.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        st.success(f"**Prediction: Tiger 🐅** (Confidence: {prediction * 100:.2f}%)")
    else:
        st.success(f"**Prediction: Leopard 🐆** (Confidence: {(1 - prediction) * 100:.2f}%)")
