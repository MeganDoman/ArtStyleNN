# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 02:03:07 2022

@author: Megan Doman
"""
from os import path
from PIL import Image
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, MaxPool2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None 

artpath = "./wikiart/wikiart/"

artworks = []
testworks = []
styles = []
teststyles = []
X_train = []
y_train = []
X_test = []
y_test = []

import csv
with open('./wikiart_csv/style_train.csv', newline='') as csvfile:
     training_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
     for row in tqdm(training_reader):
         x = row[0].split(",")
         artworks.append(x[0])
         styles.append(int(x[1]))

for i in tqdm(range(len(artworks))):
    fileloc = artpath+artworks[i]
    if path.exists(fileloc):
        y_train.append(styles[i])
        image = Image.open(fileloc)
        image = image.resize((64, 64))
        X_train.append(np.array(image))

with open('./wikiart_csv/style_val.csv', newline='') as csvfile:
     testing_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
     for row in tqdm(testing_reader):
         x = row[0].split(",")
         testworks.append(x[0])
         teststyles.append(int(x[1]))

for i in tqdm(range(len(testworks))):
    fileloc = artpath+testworks[i]
    if path.exists(fileloc):
        y_test.append(teststyles[i])
        image = Image.open(fileloc)
        image = image.resize((64, 64))
        X_test.append(np.array(image))
        
X_train = np.array(X_train)
X_train = np.resize(X_train, (len(y_train), 64, 64, 3))
X_train = X_train.astype('float32') / 255.

y_train = np.array(y_train)
y_train = keras.utils.to_categorical(y_train, 27)

X_test = np.array(X_test)
X_test = np.resize(X_test, (len(y_test), 64, 64, 3))
X_test = X_test.astype('float32') / 255.

y_test = np.array(y_test)
y_test = keras.utils.to_categorical(y_test, 27)


# building a linear stack of layers with the sequential model
model = Sequential()

# convolutional layer
model.add(Conv2D(50, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', input_shape=(64, 64, 3)))

# convolutional layer
model.add(Conv2D(75, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(125, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# flatten output of conv
model.add(Flatten())

# hidden layer
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.3))
# output layer
model.add(Dense(27, activation='softmax'))

# compiling the sequential model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# training the model for 10 epochs
model.fit(X_train, y_train, batch_size=128, epochs=10, validation_data=(X_test, y_test))

model.save('artStyle.h5')

model = load_model('artStyle.h5')

loss_train = model.history.history['loss']
loss_val = model.history.history['val_loss']
epochs = range(1,11)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

loss_train = model.history.history['accuracy']
loss_val = model.history.history['val_accuracy']
epochs = range(1,11)
plt.plot(epochs, loss_train, 'g', label='Training accuracy')
plt.plot(epochs, loss_val, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()