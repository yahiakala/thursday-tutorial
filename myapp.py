import streamlit as st
from PIL import Image, ImageOps
from detect_smile_image import detect_smile
import cv2

cascade = 'haarcascade_frontalface_default.xml'
model = 'model2.h5'

st.write("""
# Hello, my name is Yahia

## Welcome to my app!

""")

uploaded_file = st.file_uploader("Upload your profile picture.",
                                 type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file)
    fixed_image = ImageOps.exif_transpose(image)
    st.image(fixed_image)
    label = detect_smile(cascade, model, fixed_image)
    st.write(label)