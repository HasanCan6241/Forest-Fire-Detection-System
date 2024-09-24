import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Forest Fire Detection System")

st.title("🌲 Forest Fire Detection System")

st.markdown("""
This app is based on a deep learning model that can detect if there is a fire in an image. ,
Please upload an image and see the result of the model.
""")

model = tf.keras.models.load_model('final_fire_detection_model.h5')

def load_and_prep_image(image, img_size=224):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = np.array(image)
    image = tf.image.resize(image, (img_size, img_size))
    image = image / 255.0  # Normalizasyon
    return np.expand_dims(image, axis=0)  # Modelin gerektirdiği şekle getirme

uploaded_file = st.file_uploader("🔍 Please upload a fire image(jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    prepared_image = load_and_prep_image(image)

    prediction = model.predict(prepared_image)

    class_names = ["🔥 Fire", "🚫 No Fire"]

    pred_class = class_names[int(prediction[0] > 0.5)]  # Binary sınıflandırma için
    st.markdown(f"### Guess: **{pred_class}**")

    if pred_class == "🔥 Fire":
        st.warning("Fire detected! Please take necessary precautions. Call 177 Forest Fire Reporting Line")
    else:
        st.success("No fire detected. You can relax.")