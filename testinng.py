from keras.models import model_from_json
import tkinter as tk
from tkinter import filedialog
from tkinter import *
import numpy as np
from keras.preprocessing import image



def on_click():
    global path2
    try :
        json_file = open('model.json','r')
        loaded_file = json_file.read()
        json_file.close()
        model = model_from_json(loaded_file)
        model.load_weights('model.weights.h5')
        
        
        labels = ['Pepper_Bell_Bacterial_spot','Pepper_Bell_Healthy','potato_early_Blight','Potato_Healthy',"Potato_Late_Blight","Tomato_Target_Spot","Tomato_Mosaic_Virus",
                  "Tomato_Yellow_Leaf_curl_Virus","Tomato_Bacterial_spot","Tomato_Early_Blight","Tomato_Healthy","Tomato_late_Blight","Tomato_Leaf_mold","Tomato_Septoria_leaf_spot",
                  "Tomato_Spider_Mites_Two_Spotted"]
        
        
        path2 = filedialog.askopenfile().name
        
        test_image = image.load_img(path2,target_size=(236,236))
        test_image = image.img_to_array(test_image)
        test_image = test_image/255.0
        test_image = np.expand_dims(test_image,axis=0)
        
        result = model.predict(test_image)
        
        arr = np.array(result)
        max_prob = arr.argmax()
        
        prediction = labels[max_prob]
        
        lbl.configure(text=prediction)
        
    except IOError :
        print("Error")




win = tk.Tk()

lbl = Label(win,text="                          ",fg="Black")
lbl.pack()

label1 = Label(win,text="Leaf Disease Detection using GUI",fg='blue')
label1.pack()

b1 = Button(win,text="browse Image",fg="red",command=on_click)
b1.pack()

win.title("Disease detection")
win.geometry("550x250")
win.mainloop()