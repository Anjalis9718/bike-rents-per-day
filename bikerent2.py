# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 15:46:20 2019

@author: Anjali
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
data = pd.read_csv('day.csv')
dataset = data.sample(frac=1)
X = dataset.iloc[:, [2,3,4,5,6,7,8,9,10,11,12]].values
y = dataset.iloc[:, 15:16].values


from sklearn.preprocessing import LabelEncoder,OneHotEncoder

onehotencoder = OneHotEncoder(categorical_features = [2,3,4,5,6,7,8])
X = onehotencoder.fit_transform(X).toarray()


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 50, random_state = 0)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

accuracy = regressor.score(X_test,y_test)
print(accuracy)


X=np.append(arr=np.ones((731,1)).astype(int),values=X,axis=1)
import statsmodels.formula.api as sm
X_opt=X[:,[0,1,2,3,4,5,6,7,8]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()



X_opt=X[:,[0,1,2,3,4,5,7,8]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()