import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import requests
import tempfile
import os
import gdown

@st.cache_resource

def load_model():
    model_path = "models/animalclassification.h5"
    if not os.path.exists(model_path):
        url = "https://drive.google.com/file/d/1LihXDODvxvYakqDkMEAlUGjHdUXUC9y5/view?usp=sharing"
        gdown.download(url, model_path, quiet=False)
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

st.title("Animals Classifier \n Can identify these animals : butterfly, cat , dog , " \
"horse")
st.write("Upload an image !")

uploaded_file = st.file_uploader("Choose an image : ", type = ["jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image , caption = "Uploaded Image" , use_column_width = True)

    img_resized = image.resize((256, 256)) 
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array , axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    ans = predicted_class

    if ans == 0:
        print("The image is of a butterfly")
    elif ans == 1:
        print("The image is of a Cat")
    elif ans == 2:
        print("The image is of a dog")
    elif ans == 3:
        print("The image is of a horse")