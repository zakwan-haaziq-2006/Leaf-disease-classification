from keras.models import model_from_json
import numpy as np
from keras.preprocessing import image
from keras.models import load_model

json_file = open("model.json",'r')
loaded = json_file.read()
json_file.close()
model = model_from_json(loaded)
model.load_weights('model.weights.h5')
print("Model loaded Successfully...")

# model = load_model("model.weights.h5")

def Classify(img_file):
    img = img_file
    test_img = image.load_img(img,target_size= (128,128))
    test_img = image.img_to_array(test_img)
    test_img = test_img/255.0
    test_img = np.expand_dims(test_img,axis=0)
    
    result = model.predict(test_img)
    
    
    labels = ['Pepper_Bell_Bacterial_spot','Pepper_Bell_Healthy','potato_early_Blight',"Potato_Late_Blight",'Potato_Healthy',"Tomato_Bacterial_spot",
              "Tomato_Early_Blight","Tomato_Late_Blight","Tomato_Leaf_Mold",
                  "Tomato_Septorial_Leaf_spot","Tomato_Spider_Mites_Two_Spotted_Spider_mite","Tomato_Target_spot","Tomato_Yellow_leaf_curl_virus",
                  "Tomato_Mosaic_Virus",
                  "Tomato_Healthy"]
    
    #{'Pepper__bell___Bacterial_spot': 0, 'Pepper__bell___healthy': 1, 'Potato___Early_blight': 2, 'Potato___Late_blight': 3, 'Potato___healthy': 4,
    # 'Tomato_Bacterial_spot': 5, 'Tomato_Early_blight': 6, 'Tomato_Late_blight': 7, 'Tomato_Leaf_Mold': 8,
    # 'Tomato_Septoria_leaf_spot': 9, 'Tomato_Spider_mites_Two_spotted_spider_mite': 10,
    # 'Tomato__Target_Spot': 11, 'Tomato__Tomato_YellowLeaf__Curl_Virus': 12, 'Tomato__Tomato_mosaic_virus': 13, 'Tomato_healthy': 14}
    arr = np.array(result)
    max_val = arr.argmax()
    prediction = labels[max_val]
    
    print(prediction,":",img)
    

    
import os
files = []
path = "E:/Artificial Intelligence 30 days/Projects/Leaf disease classification/Dataset/PlantVillage/check"

for r,d,f in os.walk(path):
    for file in f :
        files.append(os.path.join(r,file))
        
for f in files :
    Classify(f)
    print()