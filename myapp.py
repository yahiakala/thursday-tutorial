import streamlit as st
from PIL import Image, ImageOps


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