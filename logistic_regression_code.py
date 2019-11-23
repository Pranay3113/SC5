# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 15:20:43 2019

@author: dell
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


credit_data = pd.read_csv("file:///D:/NMIMS/SEM-3/SC-5/credit.csv")

le = preprocessing.LabelEncoder()

for i in range(len(credit_data.columns)):
    if isinstance(credit_data.iloc[:,i],object)==True:
        credit_data.iloc[:,i] = le.fit_transform(credit_data.iloc[:,i])


credit_data.columns
x = credit_data.loc[:,credit_data.columns!="default"]
y = credit_data.loc[:,credit_data.columns=="default"]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.25, random_state = 0)

classfier = LogisticRegression(random_state=0)
classfier.fit(xtrain, ytrain)

y_pred = classfier.predict(xtest)

confusion_matrix(ytest, y_pred)

accuracy_score(ytest,y_pred)



