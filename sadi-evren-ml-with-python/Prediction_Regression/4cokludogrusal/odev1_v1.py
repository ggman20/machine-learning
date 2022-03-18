# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 15:41:44 2021

@author: proje7
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%%Veri onisleme
veriler = pd.read_csv('odev_tenis.csv')
temp = veriler.iloc[:,1:3].values
#%%ENCODERLAR
#Encoder : Kategorik --> Numeric OUTLOOK İÇİN

outlook = veriler.iloc[:,0:1].values
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
outlook[:,0] = le.fit_transform(veriler.iloc[:,0])
ohe = preprocessing.OneHotEncoder()
outlook = ohe.fit_transform(outlook).toarray()

#Encoder : Kategorik --> Numeric WINDY İÇİN
windy = veriler.iloc[:,3:4].values
print(windy)
windy[:,0] = le.fit_transform(windy[:,0])
print(windy)

#Encoder : Kategorik --> Numeric PLAY İÇİN
play = veriler.iloc[:,-1:].values
print(play)
le = preprocessing.LabelEncoder()
play[:,0] = le.fit_transform(play[:,0])
print(play)


#%%Numpy dizilerinin dataframe donusumu
sonuc1 = pd.DataFrame(data=outlook, index = range(14), columns = ['sunny','overcast','rainy'])
sonuc2 = pd.DataFrame(data=windy, index = range(14), columns = ['windy'])
sonuc3 = pd.DataFrame(data = play, index = range(14), columns = ['play'])
sonuc4 = pd.DataFrame(data = temp, index = range(14), columns =['Temperature','Humidity'])
#%%Dateframeleri birleştirme işlemi
s=pd.concat([sonuc1,sonuc2], axis = 1)
s2=pd.concat([s,sonuc3], axis = 1)
s3 = pd.concat([s2, sonuc4], axis = 1)
#%%Verilerin eğitim ve test şeklinde bölünmesi
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(s3,sonuc3,test_size=0.33, random_state=0)
#%%Linear Regresyon
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)


humidity = s3.iloc[:,-1:].values
sol = s3.iloc[:,:6]
x_train, x_test,y_train,y_test = train_test_split(sol,humidity,test_size=0.33, random_state=0)
r2 = LinearRegression()
r2.fit(x_train,y_train)
y_pred = r2.predict(x_test)




