# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 09:47:35 2021

@author: proje7
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%%Veri onisleme
veriler = pd.read_csv('odev_tenis.csv')
#%%ENCODERLAR
from sklearn.preprocessing import LabelEncoder
veriler2 = veriler.apply(LabelEncoder().fit_transform)
outlook = veriler2.iloc[:,:1]
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
outlook = ohe.fit_transform(outlook).toarray()
print(outlook)
#%%OneHotEncode edilmiş hava durumu bilgilerini DataFrameye dönüştür
havadurumu = pd.DataFrame(data = outlook, index = range(14), columns = ['o', 'r', 's' ])
temphumidity = veriler.iloc[:,1:3] 
windyplay = veriler2.iloc[:,-2:]
#%%Dateframeleri birleştirme işlemi
sonveriler = pd.concat([havadurumu, veriler.iloc[:,1:3]], axis = 1) 
sonveriler = pd.concat([veriler2.iloc[:,-2:], sonveriler], axis = 1)   
#%%Verilerin eğitim ve test şeklinde bölünmesi
from sklearn.model_selection import train_test_split
bagimsizdeg = sonveriler.iloc[:,:-1] #Son kolona kadar al
bagimlideg = sonveriler.iloc[:,-1:] #Sadece son kolonu al (Humidity)
x_train, x_test,y_train,y_test = train_test_split(bagimsizdeg,bagimlideg,test_size=0.33, random_state=0)
#%%Linear Regresyon
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
#%%Backward Elimination
import statsmodels.api as sm

X = np.append(arr = np.ones((14,1)).astype(int), values=bagimsizdeg, axis=1)
X_l = sonveriler.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(bagimlideg,X_l).fit()
print(model.summary())

sonveriler = sonveriler.iloc[:,1:]

X = np.append(arr = np.ones((14,1)).astype(int), values=bagimsizdeg, axis=1)
X_l = sonveriler.iloc[:,[0,1,2,3,4]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(bagimlideg,X_l).fit()
print(model.summary())


x_train = x_train.iloc[:,1:]
y_train = y_train.iloc[:,1:]
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)

x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:]
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)

