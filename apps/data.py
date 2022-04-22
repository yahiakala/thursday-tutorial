import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import imutils
import matplotlib.pyplot as plt
import cv2



def app():
    st.title("Data Analysis")
    
    st.write("""
    ## How Images are used in a machine learning model
    
    Images are input into models as arrays of numbers
    that represent color and brightness and so on. ...
    """)
    
    imgres = st.slider('Pick the resolution of the image (number of pixels)', 5, 28)
    image1 = Image.open('SMILEs/positives/positives7/10007.jpg')
    image1b = imutils.resize(np.array(image1), width=imgres)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write('This is the original image')
        st.image(image1, use_column_width=True)
    with col2:
        st.write('This is the lower resolution image')
        st.image(image1b, use_column_width=True)
    
    fig, ax = plt.subplots()
    ax.matshow(image1b, cmap=plt.cm.Blues)
    for i in range(np.size(image1b, 1)):
        for j in range(np.size(image1b, 0)):
            c = image1b[j, i]
            ax.text(i, j, str(c), va='center', ha='center', fontsize=5)
    st.write(fig)
    
    uploaded_file = st.file_uploader('Try this with your own image!',
                                     type=['jpg', 'jpeg', 'png'])
    if uploaded_file:
        image = Image.open(uploaded_file)
        fixed_image = ImageOps.exif_transpose(image) # for mobile
        gray = cv2.cvtColor(np.array(fixed_image), cv2.COLOR_BGR2GRAY)
        
        st.image(fixed_image, use_column_width=True)
        st.write('This is the lower resolution version of the image')
        fixed_imageb = imutils.resize(gray, width=imgres)
        st.image(fixed_imageb, use_column_width=True)
        
        st.write('This is more like what the algorithm sees')
        fig2, ax2 = plt.subplots()
        ax2.matshow(fixed_imageb, cmap=plt.cm.Blues)
        for i in range(np.size(fixed_imageb, 1)):
            for j in range(np.size(fixed_imageb, 0)):
                c = fixed_imageb[j, i]
                ax2.text(i, j, str(c), va='center', ha='center', fontsize=5)
        st.write(fig2)