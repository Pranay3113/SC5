# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 19:13:00 2019

@author: dell
"""

import pandas as pd
import numpy as np

data1 = pd.read_csv("file:///D:/NMIMS/SEM-3/SC-4/lastfm-matrix-germany.csv")

data1.shape

list(data1)

data1.info()

data1.isnull().sum()

data1.head()
data1.tail()

data1.describe(include='all')

data1['sum1'] = data1.iloc[:,1::].sum(axis = 1)

list(data1)

data1 = data1.sort_values(by = ['sum1'], ascending= False)
data1.head()

rating_data = data1.loc[:,data1.columns != 'user']

type(data1)

a = (data1['a perfect circle']*data1['ac/dc'])
a1 = data1['a perfect circle']**2
b1 = data1['ac/dc']**2

cos_sim = a.sum()/np.sqrt(a1.sum()*b1.sum())
rating_data.columns


import timeit as t

start = t.timeit()
sim1 = pd.DataFrame(index= rating_data.columns, columns= rating_data.columns)
for i in range(0,len(data1.columns)-1):
    for j in range(0,len(data1.columns)-1):
        a = rating_data.iloc[:,i]*rating_data.iloc[:,j]
        a1 = rating_data.iloc[:,i]**2
        b1 = rating_data.iloc[:,j]**2
        sim1.iloc[i,j] = a.sum()/np.sqrt(a1.sum()*b1.sum())
end=t.timeit()
print(end - start)

########Alternative way to calculate similarity
start = t.timeit()
sim2 = []
for i in range(0, len(data1.columns)-1):
    for i in range(0, len(data1.columns)-1):
        a = rating_data.iloc[:,i]*rating_data.iloc[:,j]
        a1 = rating_data.iloc[:,i]**2
        b1 = rating_data.iloc[:,j]**2
        cos = a.sum()/np.sqrt(a1.sum()*b1.sum())
        sim2.append(cos)
end = t.timeit()
print(end-start)

