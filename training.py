from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import BatchNormalization
from keras.callbacks import ReduceLROnPlateau

model = Sequential()


model.add(Conv2D(32,(3,3),activation='relu',input_shape = (128,128,3)))
model.add(MaxPool2D((2,2)))
model.add(BatchNormalization())


model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(BatchNormalization())

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(BatchNormalization())

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(BatchNormalization())





model.add(Flatten())

model.add(Dense(32,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(15,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


train_datagen = ImageDataGenerator(rescale = 1./255,zoom_range = 0.2,shear_range = 0.2,horizontal_flip = True)

val_datagen = ImageDataGenerator(rescale = 1./255,zoom_range = 0.2)


training_set = train_datagen.flow_from_directory("Dataset/PlantVillage/Train",target_size = (128,128),class_mode = "categorical",)
label_train = training_set.class_indices
print(label_train)
val_set = val_datagen.flow_from_directory("Dataset/PlantVillage/Val",target_size = (128,128),class_mode = 'categorical')
label_val = val_set.class_indices
print(label_val)

call_back = [EarlyStopping(monitor='val_loss',patience=5),ModelCheckpoint(filepath='model.weights.h5',monitor='val_loss',save_best_only=True,verbose=1)]


history = model.fit(training_set,steps_per_epoch=100,epochs=15,validation_data=val_set,validation_steps=2,callbacks=call_back)


import matplotlib.pyplot as plt
plt.figure(0)
plt.plot(history.history['accuracy'],label = "Accuracy")
plt.plot(history.history['val_accuracy'],label = "Validation Accuracy")
plt.title("Accuracy")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()


plt.figure(1)
plt.plot(history.history['loss'],label = "Loss")
plt.plot(history.history['val_loss'],label = "Validation Loss")
plt.title("Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()


model_json = model.to_json()
with open("model.json",'w') as json_file :
    json_file.write(model_json)
model.save_weights("model.weights.h5")
print("Model saved to disk.....")


