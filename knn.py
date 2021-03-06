# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 09:36:48 2019

@author: student
"""

#Importing Libraries

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd


#Importing the Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data" 
# Assign colum names to the dataset 
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class'] 
# Read dataset to pandas dataframe 
dataset = pd.read_csv(url, names=names)

# To see what the dataset actually looks like, execute the following command:
dataset.head()

# Preprocessing
X = dataset.drop("Class",1) 
y = dataset[["Class"]]

# Train Test Split
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Feature Scaling
from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler() 
scaler.fit(X_train) 
X_train = scaler.fit_transform(X_train) 
X_test = scaler.transform(X_test)

# Training and Predictions
from sklearn.neighbors import KNeighborsClassifier 
classifier = KNeighborsClassifier(n_neighbors=5) 
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

#Evaluating the Algorithm
from sklearn.metrics import classification_report, confusion_matrix 
print(confusion_matrix(y_test, y_pred)) 
print(classification_report(y_test, y_pred))

#Comparing Error Rate with the K Value
error = [] 
# Calculating error for K values between 1 and 40 
for i in range(1, 11): 
    knn = KNeighborsClassifier(n_neighbors=i) 
    knn.fit(X_train, y_train) 
    pred_i = knn.predict(X_test) 
    error.append(np.mean(pred_i != y_test['Class']))

plt.figure(figsize=(12, 6)) 
plt.plot(range(1, 11), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10) 
plt.title('Error Rate K Value') 
plt.xlabel('K Value') 
plt.ylabel('Mean Error')
