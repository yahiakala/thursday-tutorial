# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import imutils
import argparse
import cv2

def detect_smile(cascade, model, picture):
    """
    Detect a smile in an image.
    
    Inputs
    ------
    cascade : str
    model : str
    picture: PIL image object
    
    """
    # load the face detector cascade and smile detector CNN
    detector = cv2.CascadeClassifier(cascade)
    model = load_model(model)
    frame = np.array(picture)

    # resize the fram, convert it to grayscale, and then clone the
    # orgignal frame so we draw on it later in the program
    frame = imutils.resize(frame, width=700)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameClone = frame.copy()

    rects = detector.detectMultiScale(gray,
                                      scaleFactor=1.1,
                                      minNeighbors=5,
                                      minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)

    labellist = []
    for (fX, fY, fW, fH) in rects:
        # extract the ROI of the face from the grayscale image
        # resize it to a fixed 28x28 pixels, and then prepare the
        # ROI for classification via the CNN
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (28, 28))
        roi = roi.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)  # add a dummy dimension
        (notSmiling, Smiling) = model.predict(roi)[0]
        label = 'Smiling' if Smiling > notSmiling else "Not Smiling"
    labellist.append(label)
    label2 = 'Smiling' if 'Smiling' in labellist else 'Not Smiling'
    return label2


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--cascade', required=True, 
        help='path to where the face cascade resides')
    ap.add_argument('-m', '--model', required=True, 
        help='path to the pre-trained smile detector CNN')
    ap.add_argument('-p', '--picture', 
        help='path to the image file')
    args = vars(ap.parse_args())
    label2 = detect_smile(args['cascade'],
                          args['model'], args['picture'])
    print(label2)