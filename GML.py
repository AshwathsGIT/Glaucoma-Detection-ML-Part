#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 14:30:42 2018

@author: ashwathsridhar
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets, linear_model


ds= pd.read_csv('GDS.csv')
X = ds.iloc[:,1].values
Y = ds.iloc[:,0].values

from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size = 0.9 ,random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train.reshape(-1,1), Y_train.reshape(-1,1))

inp = 0.30
Y_pred = regressor.predict(inp)
#print(X_test)
print(Y_pred)
  

# Train set graph
"""plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train.reshape(-1,1)),color = 'blue')
plt.title('Severity vs CDR')
plt.xlabel('CDR')
plt.ylabel('Severity')
plt.show()

#test set graph
plt.scatter(X_test, inp, color = 'red')
plt.plot(X_train, regressor.predict(inp),color = 'blue')
plt.title('Severity vs CDR')
plt.xlabel('CDR')
plt.ylabel('Severity')
plt.show()"""