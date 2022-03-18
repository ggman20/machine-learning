# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 13:22:28 2022

@author: ARMAN
"""
#https://sogrekci.com/ders-notu/derin-ogrenme/4-mnist-veri-seti/

import tensorflow as tf
#%%
x_train, y_train = tf.keras.datasets.mnist.load_data()[0]
print(x_train.shape)
print(y_train.shape)

print(x_train[1000])
print(x_train[1000].shape)

print(y_train[1000])

for row in x_train[1000]:
    for i in row:
        print("%3s "%i, end='')
    print()
    
import matplotlib.pyplot as plt
plt.imshow(x_train[1000], cmap='gray_r')
plt.show()

import numpy as np

n, k, figsize = 10, 10, (10, 10)
fig, ax = plt.subplots(n, k, figsize=figsize)
for i in range(n):
    for j in range(k):
        ax[i,j].imshow(x_train[np.random.randint(x_train.shape[0])], cmap='gray_r')
        ax[i,j].axis('off')

plt.show()

img1000 = x_train[1000].reshape(28*28,)

for px in img1000:
    print("%s "%px, end='')