# import the Package
import os
import cv2
import numpy as np 
from PIL import Image , ImageOps
import streamlit as st
from tensorflow.keras.models import load_model
import tensorflow as tf

class_name = ["airplane", "automobile" , "bird" , "cat" , "deer" , "dog" , "frog" , "horse" , "ship" , "truck"]

# Create a function to load my saved model
@st.cache(allow_output_mutation=True)
def load_my_model():
    model = tf.keras.models.load_model("D:\Keras_Tutorial\my_model.h5")
    return model

model = load_my_model()

# Create a title of web App
st.title("Image Classification with Cifar10 Dataset")
st.header("Please Upload images related to this things...")
st.text(class_name)

# create a file uploader and take a image as an jpg or png
file = st.file_uploader("Upload the image" , type=["jpg" , "png"])

# Create a function to take and image and predict the class
def import_and_predict(image_data , model):
    size = (32 ,32)
    image = ImageOps.fit(image_data , size , Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis,...]
    prediction = model.predict(img_reshape)
    return prediction

if st.button("Predict"):
    image = Image.open(file)
    st.image(image , use_column_width=True)
    predictions = import_and_predict(image , model)

    class_name = ["airplane", "automobile" , "bird" , "cat" , "deer" , "dog" , "frog" , "horse" , "ship" , "truck"]

    string = "Image mostly same as :-" + class_name[np.argmax(predictions)]
    st.success(string)
































# # create a function for model prediction
# def predict_image(image_path , index):
#     model = load_model("D:\Keras_Tutorial\HandsOn_Keras\model_cifar.h5")
#     image_path = image_path / 255
#     y_pred = model.predict(image_path)
#     y_classes = [np.argmax(element) for element in y_pred]
#     y_pred = st.write(classes[y_classes[index]])
#     return y_pred

# # Classes names
# classes = ["airplane", "automobile" , "bird" , "cat" , "deer" , "dog" , "frog" , "horse" , "ship" , "truck"]

# st.title("Image Classification using Cifar10 Datasets")
# upload_file = st.file_uploader("Choose an image..." , type = "jpg")

# def predict_button():
#     if upload_file is not None:
#         image = Image.open(upload_file)
#         st.image(image , caption="Uploaded Image." , use_column_width = "True")
#         st.write("")
#         st.write("Classifying")
#         y_classes = predict_image(upload_file , index = model.index)

# if st.button("Predict"):
#     predict_button()






    