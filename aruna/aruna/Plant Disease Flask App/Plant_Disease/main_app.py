#Library imports
import requests
import numpy as np
import streamlit as st
from streamlit_lottie import st_lottie
import cv2
from keras.models import load_model


#Loading the Model
model = load_model('C:/Users/Aditya Aggarwal/Downloads/aruna/aruna/Plant Disease Flask App/Plant_Disease/plant_disease.h5')

#Name of Classes
CLASS_NAMES = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']

def load_lottie(url):
    r=requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_coding =load_lottie("https://assets3.lottiefiles.com/private_files/lf30_j4v2bg0q.json")
#Setting Title of App
with st.container():
    left_column,right_column=st.columns(2)
    with left_column:
        st.title("Plant Disease Detection")
        st.markdown("Upload an image of the plant leaf")

#Uploading the dog image
        plant_image = st.file_uploader("Choose an image...", type="jpg")
        submit = st.button('Predict')
#On predict button click
        if submit:


            if plant_image is not None:

        # Convert the file to an opencv image.
                file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
                opencv_image = cv2.imdecode(file_bytes, 1)



        # Displaying the image
                st.image(opencv_image, channels="BGR")
                st.write(opencv_image.shape)
        #Resizing the image
                opencv_image = cv2.resize(opencv_image, (256,256))
        #Convert image to 4 Dimension
                opencv_image.shape = (1,256,256,3)
        #Make Prediction
                Y_pred = model.predict(opencv_image)
                result = CLASS_NAMES[np.argmax(Y_pred)]
                st.title(str("This is "+result.split('-')[0]+ " leaf with " + result.split('-')[1]))

    with right_column:
        st_lottie(lottie_coding, height=500, key="Plant")
