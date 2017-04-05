"""
A baseline model test
Python version: Python 3.5.2
Date: 21.03.2015
Packages: sklearn: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model
          pandas: http://pandas.pydata.org/pandas-docs/stable/io.html#io-store-in-csv
linear regression models: http://scikit-learn.org/stable/modules/linear_model.html
"""
"""
BayesianRegression Model
"""
# import pandas as pd   # for import and export the data to csv
# from sklearn.linear_model import LinearRegression as lr
# from sklearn.metrics import mean_squared_error as mse
# from sklearn.metrics import make_scorer
# from sklearn.model_selection import cross_val_score
# import numpy as np
# from sklearn.linear_model import BayesianRidge
#
# def root_mean_squared_error(y_true, y_pred, sample_weight=None, multioutput='uniform_average'):
#     rmse = np.sqrt(mse(y_true, y_pred,
#                        sample_weight=sample_weight, multioutput=multioutput))
#     return rmse
#
# # import data
# # for training
# rawDataTrain = pd.read_csv('train.csv').values
#
# X_train = rawDataTrain[:, 2:]   # the features
# y_train = rawDataTrain[:, 1]   # the target value
#
# # for test
# rawDataTest = pd.read_csv('test.csv').values
# X_test = rawDataTest[:, 1:]
# y_test = np.zeros([X_test.shape[0], 2])   # [id, y]
# y_test[:, 0] = rawDataTest[:, 0]
#
# # linear regression
# clf = BayesianRidge(compute_score=True)
# clf.fit(X_train, y_train)
#
# cvScore = cross_val_score(clf,
#                           X_train, y_train,
#                           scoring=make_scorer(root_mean_squared_error),
#                           verbose=0,
#                           cv=100)
# print('Mean of CV RMSE:', np.mean(cvScore))
#
# y_test[:, 1] = clf.predict(X_test)
# df = pd.DataFrame(data=y_test)
# df.to_csv('./bayesianridge_cv=100.csv', header=['Id', 'y'], sep=',', index=False)


"""
Ridge
"""
# import pandas as pd   # for import and export the data to csv
# from sklearn.metrics import mean_squared_error as mse
# from sklearn.metrics import make_scorer
# from sklearn.model_selection import cross_val_score
# import numpy as np
# from sklearn.linear_model import Ridge
#
# def root_mean_squared_error(y_true, y_pred, sample_weight=None, multioutput='uniform_average'):
#     rmse = np.sqrt(mse(y_true, y_pred,
#                        sample_weight=sample_weight, multioutput=multioutput))
#     return rmse
#
# # import data
# # for training
# rawDataTrain = pd.read_csv('train.csv').values
#
# X_train = rawDataTrain[:, 2:]   # the features
# y_train = rawDataTrain[:, 1]   # the target value
#
# # for test
# rawDataTest = pd.read_csv('test.csv').values
# X_test = rawDataTest[:, 1:]
# y_test = np.zeros([X_test.shape[0], 2])   # [id, y]
# y_test[:, 0] = rawDataTest[:, 0]
#
# # linear regression
# clf=Ridge(alpha=0.1)
# clf.fit(X_train,y_train)
#
# cvScore = cross_val_score(clf,
#                           X_train, y_train,
#                           scoring=make_scorer(root_mean_squared_error),
#                           verbose=0,
#                           cv=100)
# print('Mean of CV RMSE:', np.mean(cvScore))
#
# y_test[:, 1] = clf.predict(X_test)
# df = pd.DataFrame(data=y_test)
# df.to_csv('./Ridge_cv=100.csv', header=['Id', 'y'], sep=',', index=False)


"""
HuberRegression
"""
# import pandas as pd   # for import and export the data to csv
# from sklearn.metrics import mean_squared_error as mse
# from sklearn.metrics import make_scorer
# from sklearn.model_selection import cross_val_score
# import numpy as np
# from sklearn.linear_model import HuberRegressor
#
# def root_mean_squared_error(y_true, y_pred, sample_weight=None, multioutput='uniform_average'):
#     rmse = np.sqrt(mse(y_true, y_pred,
#                        sample_weight=sample_weight, multioutput=multioutput))
#     return rmse
#
#
#
# # import data
# # for training
# rawDataTrain = pd.read_csv('train.csv').values
#
# X_train = rawDataTrain[:, 2:]   # the features
# y_train = rawDataTrain[:, 1]   # the target value
#
# # for test
# rawDataTest = pd.read_csv('test.csv').values
# X_test = rawDataTest[:, 1:]
# y_test = np.zeros([X_test.shape[0], 2])   # [id, y]
# y_test[:, 0] = rawDataTest[:, 0]
#
# # linear regression
# clf=HuberRegressor(fit_intercept=False, alpha=0, max_iter=1000)
# clf.fit(X_train,y_train)
#
# cvScore = cross_val_score(clf,
#                           X_train, y_train,
#                           scoring=make_scorer(root_mean_squared_error),
#                           verbose=0,
#                           cv=100)
# print('Mean of CV RMSE:', np.mean(cvScore))
#
# y_test[:, 1] = clf.predict(X_test)
# df = pd.DataFrame(data=y_test)
# df.to_csv('./Huberreg_cv=100.csv', header=['Id', 'y'], sep=',', index=False)



"""
Lars lasso Regression
"""
import pandas as pd   # for import and export the data to csv
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.linear_model import LassoLars

def root_mean_squared_error(y_true, y_pred, sample_weight=None, multioutput='uniform_average'):
    rmse = np.sqrt(mse(y_true, y_pred,
                       sample_weight=sample_weight, multioutput=multioutput))
    return rmse



# import data
# for training
rawDataTrain = pd.read_csv('train.csv').values

X_train = rawDataTrain[:, 2:]   # the features
y_train = rawDataTrain[:, 1]   # the target value

# for test
rawDataTest = pd.read_csv('test.csv').values
X_test = rawDataTest[:, 1:]
y_test = np.zeros([X_test.shape[0], 2])   # [id, y]
y_test[:, 0] = rawDataTest[:, 0]

# linear regression
clf=LassoLars(alpha=0.00000001)
clf.fit(X_train,y_train)

cvScore = cross_val_score(clf,
                          X_train, y_train,
                          scoring=make_scorer(root_mean_squared_error),
                          verbose=0,
                          cv=100)
print('Mean of CV RMSE:', np.mean(cvScore))

y_test[:, 1] = clf.predict(X_test)
df = pd.DataFrame(data=y_test)
df.to_csv('./Laslasso_cv=100.csv', header=['Id', 'y'], sep=',', index=False)