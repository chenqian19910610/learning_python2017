"""
Packages: sklearn: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model
          pandas: http://pandas.pydata.org/pandas-docs/stable/io.html#io-store-in-csv
LassoCV: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html#examples-using-sklearn-preprocessing-polynomialfeatures
Polinomial features:http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html#examples-using-sklearn-preprocessing-polynomialfeatures
"""
import pandas as pd   # for import and export the data to csv
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV


def root_mean_squared_error(y_true, y_pred, sample_weight=None, multioutput='uniform_average'):
    rmse = np.sqrt(mse(y_true, y_pred,
                       sample_weight=sample_weight, multioutput=multioutput))
    return rmse

# import data
# for training
rawDataTrain = pd.read_csv('train.csv').values
X_train = rawDataTrain[:, 2:]   # the features
y_train = rawDataTrain[:, 1]   # the target value
X_train = PolynomialFeatures(degree=3).fit_transform(X_train)


# for test
rawDataTest = pd.read_csv('test.csv').values
X_test = rawDataTest[:, 1:]
X_test = PolynomialFeatures(degree=3).fit_transform(X_test)
y_test = np.zeros([X_test.shape[0], 2])   # [id, y]
y_test[:, 0] = rawDataTest[:, 0]

# Lassocv
clf = LassoCV(cv=10)
clf.fit(X_train, y_train)

cvScore = cross_val_score(clf,
                          X_train, y_train,
                          scoring=make_scorer(root_mean_squared_error),
                          verbose=0,
                          cv=10)
print('Mean of CV RMSE:', np.mean(cvScore))

y_test[:, 1] = clf.predict(X_test)
df = pd.DataFrame(data=y_test)
df.to_csv('./lassocv.csv', header=['Id', 'y'], sep=',', index=False)
