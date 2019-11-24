# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 12:33:42 2019

@author: student
"""

import pandas as pd
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
import numpy as np

data= pd.read_csv("C:/Users/HP/Downloads/spam.csv",encoding='latin1')

data.columns

label=data['v1']
features=data['V2']

#Count plot in seaborn
ax=sns.countplot(x='v1',data = data)

from sklearn.feature_extraction.text import TfidfVectorizer
tv = TfidfVectorizer(max_features = 5000)
features = tv.fit_transform(features).toarray()

encoder = LabelEncoder()
y = encoder.fit_transform(label)

features_train, features_test, label_train, label_test = train_test_split(features, y, test_size = .10, random_state = 0)

gnb = GaussianNB()
gnb.fit(features_train, label_train)
print(gnb.score(features_train, label_train))  
print(gnb.score(features_test, label_test))    


nb = MultinomialNB()
nb.fit(features_train, label_train)
print("accuracy:", nb.score(features_test, label_test))
