import streamlit as st
from PIL import Image


st.write("""
# Hello, my name is Yahia

## Welcome to my app!

""")

uploaded_file = st.file_uploader("Upload your profile picture.",
                                 type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    st.image(uploaded_file)