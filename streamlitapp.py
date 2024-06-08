# Import necessary libraries
import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# Define helper functions
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def resnet_predict(img_path):
    # Load pre-trained ResNet50 model
    model = load_model('resnet_model.h5')

    # Preprocess image
    x = preprocess_image(img_path)

    # Make prediction
    preds = model.predict(x)
    return preds

def detect_tumor(img_path):
    # Load image and convert to grayscale
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Perform edge detection using Canny algorithm
    edges = cv2.Canny(gray, 100, 200)

    # Apply morphological operations to remove noise and fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Find contours of tumor region
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # Draw bounding box around tumor region
        x,y,w,h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

    # Save image with bounding box
    cv2.imwrite('output.png', img)

# Set up Streamlit app
st.set_page_config(page_title='Esophageal Cancer Detection', page_icon=':microscope:')

st.title('Esophageal Cancer Detection')

# Upload image file
uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Save uploaded file to disk
    with open('temp.jpg', 'wb') as f:
        f.write(uploaded_file.read())

    # Display uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Perform tumor detection and display output image
    detect_tumor('temp.jpg')
    st.image('output.png', caption='Output Image', use_column_width=True)

    # Perform cancer classification using ResNet50 and display prediction
    resnet_preds = resnet_predict('temp.jpg')
    top_pred = decode_predictions(resnet_preds, top=1)[0][0]
    st.write(f"ResNet50 Prediction: {top_pred[1]} ({round(top_pred[2]*100, 2)}%)")
