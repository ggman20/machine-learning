# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 13:44:01 2022

@author: ARMAN
"""
#https://ichi.pro/tr/mnist-i-anlama-ve-mnist-ve-fashion-mnist-veri-kumeleri-ile-siniflandirma-modeli-olusturma-130530865218743

import tensorflow as tf
from tensorflow import keras

mnist = keras.datasets.mnist # replace mnist with fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data() # replace mnist with fashion_mnist

# Splitting data set
X_valid, X_train = X_train_full[ :5000] / 255, X_train_full[5000: ]/255
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test/255

# printing the data at index 0
X_valid[0]

# Visulising the image at index 5
import matplotlib.pyplot as plt

plt.imshow(X_train[5], cmap = 'binary') # change 5 to any other index value between 0 and 4999 to visualise the image
plt.axis('off')
plt.show

class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

#Checking the class/label of image at index 2
class_names[y_train[2]] # change the index value of 2 to play around

# Visualising the dataset
n_rows = 4
n_cols = 10
plt.figure(figsize=(n_cols * 1.2, n_rows * 1.2))
for row in range(n_rows):
    for col in range(n_cols):
        index = n_cols * row + col
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(X_train[index], cmap="binary", interpolation="nearest")
        plt.axis('off')
        plt.title(class_names[y_train[index]], fontsize=12)
plt.subplots_adjust(wspace=0.2, hspace=0.5)

#Setting seed value before starting with deep learning
import numpy as np

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

# Creating and setting up a Sequential Model
model = keras.models.Sequential([
  keras.layers.Flatten(input_shape=[28, 28]),
  keras.layers.Dense(300, activation='relu'),
  keras.layers.Dense(100, activation='relu'),
  keras.layers.Dense(10, activation='softmax')
])

# Compiling the model
model.compile(loss = 'sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy','mse'])

# Training the model
history = model.fit(X_train, y_train, epochs= 10, validation_data=(X_valid, y_valid)) # Change epochs to 30 if using Fashion_MNIST dataset

# Visualising loss and accuracy of trained algorithm
import pandas as pd

pd.DataFrame(history.history).plot(figsize = (8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()

# Creating a set of data from the test data which trained algorithm is unaware of
X_new = X_test[:3] #Change the size of this sample to predict the class/label of any other image from the test set.

#Predicting class/label by showing new data to the trained model. Model will try to tell what number it is.
y_pred = model.predict(X_new) 
classes_x = np.argmax(y_pred,axis=1)
# y_pred = model.predict_classes(X_new)

# Printing the predicted labels
# np.array(class_names)[y_pred]

# Viewing the printing Labels
plt.figure(figsize=(7.2, 2.4))
for index, image in enumerate(X_new):
  plt.subplot(1, 3, index + 1)
  plt.imshow(image, cmap = 'binary', interpolation= 'nearest')
  plt.axis = ('off')
  # plt.title(class_names[y_pred[index]], fontsize=12)
plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()

# Here the number you see on the image should be the number of the label. 
# What this shows is the image which model used as an input and then predict the label of the image.

