import streamlit as st
from PIL import Image
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from keras.models import model_from_json
import cv2

json = open("model.json")
loaded = json.read()
json.close()
model = model_from_json(loaded)
model.load_weights("model.weights.h5")



labels = ['Pepper_Bell_Bacterial_spot','Pepper_Bell_Healthy','potato_early_Blight',"Potato_Late_Blight",'Potato_Healthy',"Tomato_Bacterial_spot",
              "Tomato_Early_Blight","Tomato_Late_Blight","Tomato_Leaf_Mold",
                  "Tomato_Septorial_Leaf_spot","Tomato_Spider_Mites_Two_Spotted_Spider_mite","Tomato_Target_spot","Tomato_Yellow_leaf_curl_virus",
                  "Tomato_Mosaic_Virus",
                  "Tomato_Healthy"]


def preprocessor(img):
    img = img.resize(128,128)
    img_arrady = np.array(img)/255.0
    img_arrady = np.expand_dims(img_arrady,axis=0)
    
    return img_arrady


st.title("Leaf Disease Detection App")
st.write("Upload the image of the leaf")

uploaded_file = st.file_uploader("Chose a img",type=['jpeg','jpg','png'])

if uploaded_file :
    image = Image.open(uploaded_file)
    st.image(image=image,caption="UPloaded leaf image",use_column_width=True)
    processed_img = preprocessor(image)
    
    prediction = model.predict(processed_img)[0]
    
    predicted_class = labels[np.argmax(prediction)]
    confidence = np.max(prediction)
    
    st.success("Predicted diseasee",predicted_class)
    st.info("confidence :",confidence)
     