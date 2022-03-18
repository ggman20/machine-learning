# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 22:59:44 2021

@author: ARMAN
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 16:16:03 2020

@author: sadievrenseker
"""


#%%1.Kütüphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

#%%2.Veri onisleme
#2.1.Veri yukleme
veriler = pd.read_excel('Iris.xls')
print(veriler)

x = veriler.iloc[:,0:4].values #bağımsız değişkenler
y = veriler.iloc[:,4:].values #bağımlı değişken
print(y)
#2.2 Verilerde Label Encoding ile Sayılara Dönüştürme
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

y = le.fit_transform(y)

#2.3.Verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#2.4.Verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

#%%3.Classification_LR
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred = logr.predict(X_test)
print(y_pred)
print(y_test)

#%%4.Karmaşıklık Matrisi ile Karşılaştırma-Analiz
cm = confusion_matrix(y_test,y_pred)
print(cm)

#%%5.Classification_KNN
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print(cm)

#%%6.Classification_SVM
from sklearn.svm import SVC
svc = SVC(kernel='rbf')
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('SVC')
print(cm)

#%%7.Classification_NB
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('GNB')
print(cm)

#%%8.Classification_DT
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'entropy')

dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('DTC')
print(cm)

#%%8.Classification_RF
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, criterion = 'entropy')
rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('RFC')
print(cm)
    
#%%9.ROC , TPR, FPR değerleri 
y_proba = rfc.predict_proba(X_test)
print(y_proba, "y_proba")
print(y_test,"y_test")
print(y_proba[:,0])

from sklearn import metrics
fpr , tpr , thold = metrics.roc_curve(y_test,y_proba[:,0],pos_label='e')
print(fpr, "fpr")
print(tpr, "tpr")
print(thold, "threshold")








