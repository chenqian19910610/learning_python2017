"""
A baseline model test
Python version: Python 3.5.2
Date: 15.03.2015
Packages: sklearn: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model
          pandas: http://pandas.pydata.org/pandas-docs/stable/io.html#io-store-in-csv
Author: CZ
"""
import pandas as pd   # for import and export the data to csv
from sklearn.linear_model import LinearRegression as lr
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
import numpy as np
import os

def root_mean_squared_error(y_true, y_pred, sample_weight=None, multioutput='uniform_average'):
    rmse = np.sqrt(mse(y_true, y_pred,
                       sample_weight=sample_weight, multioutput=multioutput))
    return rmse

# import data
# for training
#rawDataTrain = pd.read_csv('.\LIS2017project\datasets\train.csv').values
rawDataTrain = pd.read_csv('train.csv').values

#path = os.getcwd()
#filename = path+'\data\\train.csv'
#print(filename)
#rawDataTrain = pd.read_csv(filename).values
#rawDataTrain = pd.read_csv('.\data\\train.csv').values
#print(list(rawDataTrain[2]))

X_train = rawDataTrain[:, 2:]   # the features
y_train = rawDataTrain[:, 1]   # the target value

# for test
rawDataTest = pd.read_csv('test.csv').values
X_test = rawDataTest[:, 1:]
y_test = np.zeros([X_test.shape[0], 2])   # [id, y]
y_test[:, 0] = rawDataTest[:, 0]

# linear regression
linearReg = lr()
cvScore = cross_val_score(linearReg,
                          X_train, y_train,
                          scoring=make_scorer(root_mean_squared_error),
                          verbose=0,
                          cv=10)
print('Mean of CV RMSE:', np.mean(cvScore))

linearReg.fit(X_train, y_train)
y_test[:, 1] = linearReg.predict(X_test)
df = pd.DataFrame(data=y_test)
df.to_csv('./ls.csv', header=['Id', 'y'], sep=',', index=False)
