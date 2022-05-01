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
from sklearn.metrics import confusion_matrix

#returns an error if removed, images too large
Image.MAX_IMAGE_PIXELS = None 

#WikiArt dataset
artpath = "./wikiart/wikiart/"

#names of artworks
artworks = []
testworks = []

#style by number
styles = []
teststyles = []

# vectorized images
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
y_train = keras.utils.to_categorical(y_train, 27) #or 27, depends on classes

X_test = np.array(X_test)
X_test = np.resize(X_test, (len(y_test), 64, 64, 3))
X_test = X_test.astype('float32') / 255.

y_test = np.array(y_test)
y_test = keras.utils.to_categorical(y_test, 27) #depends o classes


# building a linear stack of layers with the sequential model
model = Sequential()

# convolutional layer
model.add(Conv2D(50, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', input_shape=(64, 64, 3)))

# convolutional layers
model.add(Conv2D(75, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(125, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(200, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
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

# training the model for 20 epochs
model.fit(X_train, y_train, batch_size=128, epochs=20, validation_data=(X_test, y_test))

#best so far -- 40% validation accuracy
model.save('artStyle60.h5')

#model = load_model('artStyle5.h5')

loss_train = model.history.history['loss']
loss_val = model.history.history['val_loss']
epochs = range(1,21)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('LossTrainVTest60.png')
plt.show()

loss_train = model.history.history['accuracy']
loss_val = model.history.history['val_accuracy']
epochs = range(1,21)
plt.plot(epochs, loss_train, 'g', label='Training accuracy')
plt.plot(epochs, loss_val, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('AccuracyTrainVTest60.png')
plt.show()

from contextlib import redirect_stdout

with open('modelsummary60.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()

y_truth = np.argmax(y_test, axis=1)
predictions = model.predict(x=X_test, batch_size=128, verbose=1)
rounded_predictions = np.argmax(predictions, axis=1)

cm = confusion_matrix(y_true=y_truth, y_pred=rounded_predictions)


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('ConfusionMatrix60.png')
cm_plot_labels = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')