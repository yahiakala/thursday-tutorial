import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
from lenet.nn.conv import LeNet  # local file
from imutils import paths
import imutils
from PIL import Image
import matplotlib.pyplot as plt
import altair as alt
import numpy as np
import pandas as pd
import argparse
import cv2
import os
import io


def plot_train_results():
    fitresult_plot = pd.read_csv('fitresult.csv')
    # Matplotlib
    # fig, ax = plt.subplots()
    # ax.plot(np.arange(0, 15), fitresult_plot['loss'], label='train_loss')
    # ax.plot(np.arange(0, 15), fitresult_plot['val_loss'], label='val_loss')
    # ax.plot(np.arange(0, 15), fitresult_plot['accuracy'], label='accuracy')
    # ax.plot(np.arange(0, 15), fitresult_plot['val_accuracy'], label='val_accuracy')
    # ax.set_title('Training Loss and Accuracy')
    # ax.set_xlabel('Epoch #')
    # ax.set_ylabel('Loss/Accuracy')
    # ax.legend()
    # st.write(fig)
    
    # Show altair
    fitresult_plot['epoch'] = np.arange(1, 16)
    fitresult2 = pd.melt(fitresult_plot, id_vars=['epoch'], var_name='series', value_name='Loss/Accuracy')
    # Create a selection that chooses the nearest point & selects based on x-value
    nearest = alt.selection(type='single', nearest=True, on='mouseover',
                            fields=['epoch'], empty='none')

    # The basic line
    line = alt.Chart(fitresult2).mark_line(interpolate='basis').encode(
        x='epoch:Q',
        y='Loss/Accuracy:Q',
        color='series:N'
    )

    # Transparent selectors across the chart. This is what tells us
    # the x-value of the cursor
    selectors = alt.Chart(fitresult2).mark_point().encode(
        x='epoch:Q',
        opacity=alt.value(0),
    ).add_selection(
        nearest
    )

    # Draw points on the line, and highlight based on selection
    points = line.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )
    # Draw text labels near the points, and highlight based on selection
    text = line.mark_text(align='left', dx=5, dy=-5).encode(
        text=alt.condition(nearest, 'Loss/Accuracy:Q', alt.value(' '), format='.3f')
    )
    # Draw a rule at the location of the selection
    rules = alt.Chart(fitresult2).mark_rule(color='gray').encode(
        x='epoch:Q',
    ).transform_filter(
        nearest
    )

    # Put the five layers into a chart and bind the data
    st.altair_chart(alt.layer(
        line, selectors, points, rules, text
    ).configure_legend(
        orient='bottom'
    ), use_container_width=True)


def app():
    st.title('Training the Model')
    # Show one smile example and the pixelated version
    st.write("## Smiling Example")
    image1 = Image.open('SMILEs/positives/positives7/10007.jpg')
    image1b = imutils.resize(np.array(image1), width=28) # 28 x 28 x 1
    image2 = Image.open('SMILEs/negatives/negatives7/10001.jpg')
    image2b = imutils.resize(np.array(image2), width=28)

    
    col1, col2 = st.columns(2)
    with col1:
        st.write('Original')
        st.image(image1, use_column_width=True)
    # Show one frown example and the pixelated version
    with col2:
        st.write('Resized for model training')
        st.image(image1b, use_column_width=True)
    
    st.write("## Not Smiling Example")
    col3, col4 = st.columns(2)
    with col3:
        st.write('Original')
        st.image(image2, use_column_width=True)
    with col4:
        st.write('Resized for model training')
        st.image(image2b, use_column_width=True)
    
    block = st.container()
    with block:
        if os.path.exists('fitresult.csv'):
            st.write('## Model Training Results')
            plot_train_results()

    if st.button('Train/Retrain Model'):
        with st.spinner('This could take about 5 minutes locally, 1 minute on the web'):
            st.write('## Model Training Results')
            training_model()
        st.balloons()
        with block:
            plot_train_results()
    



def training_model():
    args = {}
    args['dataset'] = './SMILEs'
    args['model'] = './model3.h5'

    # initialize the list of data and labels
    data = []
    labels = []

    # loop over the input images
    for imagePath in sorted(list(paths.list_images(args['dataset']))):
        # load the image, pre-process it, and store in the data list
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = imutils.resize(image, width=28) # 28 x 28 x 1
        image = img_to_array(image)
        data.append(image)

        # extract the class label from the image path and update the labels list
        label = imagePath.split(os.path.sep)[-3]
        label = 'smiling' if label == 'positives' else 'not_smiling'
        labels.append(label)

    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype='float') / 255.0 # 0 to 255
    labels = np.array(labels)

    # convert the labels from integers to vectors
    le = LabelEncoder().fit(labels)
    labels = np_utils.to_categorical(le.transform(labels), 2)

    # account for skew in the labeled data
    classTotals = labels.sum(axis=0)
    classWeight = dict()

    for i in range(0, len(classTotals)):
        classWeight[i] = classTotals.max() / classTotals[i]

    # partition the data into training and testing sploits using 80% of
    # the data for training and the remaining 20% for testing
    (trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                      test_size=0.20,
                                                      stratify=labels,
                                                      random_state=42)

    # initialize the model
    print('[INFO] compiling model...')
    model = LeNet.build(width=28, height=28, depth=1, classes=2)
    model.compile(loss=['binary_crossentropy'], optimizer='adam', metrics=['accuracy'])

    # train the network
    print('[INFO] training network...')
    H = model.fit(trainX, trainY, validation_data=(testX, testY),
                  class_weight=classWeight, batch_size=64, epochs=15, verbose=1)

    # evaluate the network
    print('[INFO] evaluating network...')
    predictions = model.predict(testX, batch_size=64)
    print(classification_report(testY.argmax(axis=1),
                                predictions.argmax(axis=1),
                                target_names=le.classes_))

    # save the model to disk
    print('[INFO] serializing network')
    model.save(args['model'])
    fitresult = pd.DataFrame(H.history)
    fitresult.to_csv('fitresult.csv', index=False)