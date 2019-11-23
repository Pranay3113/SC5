# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 09:05:46 2019

@author: student
"""
import pandas as pd
import numpy as np
import scipy as sp
import statsmodels as sm
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

dta = pd.read_csv("file:///C:/Users/student/Desktop/credit.csv")
a = dta.select_dtypes(include='object')
#Import label encoder
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()

for i in a.columns:
    dta[i]= label_encoder.fit_transform(dta[i])
dta.dtypes

X = dta.drop("default",1)
Y= dta[["default"]]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)
Data = pd.DataFrame(columns = range(1,6), index = range(1,6))
q = range(1,6)

for i in range(0,len(q)):
    for j in range(0,len(q)):
        clf_entropy = DecisionTreeClassifier(criterion='entropy', random_state = 1, max_depth = q[i], min_samples_leaf = q[j])
        clf_entropy.fit(X_train,y_train)
        y_pred_test = clf_entropy.predict(X_test)
        y_pred_train = clf_entropy.predict(X_train)
        Data.iloc[i, j] = accuracy_score(y_test, y_pred_test)