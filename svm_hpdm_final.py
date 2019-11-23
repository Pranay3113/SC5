# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 10:40:41 2019

@author: student
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
Data=pd.read_csv("C:/Users/student/Desktop/msc asa 06/Iris (1).csv")
Data.columns
sns.pairplot(data=Data[['sepal length ', 'sepal width ', 'class']], hue='class', palette='Set2')

X=Data.iloc[:,:2]
y=Data[['class']]

#train_test_split
from sklearn.model_selection import train_test_split
X_train,  X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#using dictionary
c = [1, 10, 100]
dct = {}

for l in c:
    acc = pd.DataFrame(columns = ['linear', 'rbf', 'sigmoid', 'poly'], index = [0.1,10,100])
    for i in acc.index:
        for j in acc.columns:
            model=SVC(kernel = j,C = l ,gamma = i)
            model.fit(X_train, y_train)
            acc.loc[i, j] = accuracy_score(y_test, model.predict(X_test))
    dct[l] = acc