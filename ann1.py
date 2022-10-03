# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 21:04:47 2022

@author: fawad
"""



# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('dataset.csv')
X = dataset.iloc[:, [0,1,2,3]].values
y = dataset.iloc[:, -1].values



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#importing keras libraries and packages
import keras 
from keras.models import Sequential
from keras.layers import Dense

#initializing the ANN
classifier = Sequential()

#Adding the input layer and first hidden layer
classifier.add(Dense(3, kernel_initializer='random_uniform',activation='relu',input_dim=4))

#Adding the second hidden layer
classifier.add(Dense(3, kernel_initializer='random_uniform',activation='relu'))

#Adding the output layer
classifier.add(Dense(1, kernel_initializer='random_uniform',activation='sigmoid'))

#compiling the ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Fitting the ANN to the training set 
classifier.fit(X_train, y_train , batch_size=5, epochs=100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)

#saving trained model
from keras.models import load_model
classifier.save('model.h5')
mdl = load_model('model.h5')

mdl_check=mdl.predict(X_test)