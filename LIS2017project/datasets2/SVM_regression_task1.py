"""
A baseline model test
Python version: Python 3.5.2
Date: 25.03.2015
Packages: sklearn: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model
          pandas: http://pandas.pydata.org/pandas-docs/stable/io.html#io-store-in-csv
SVR: http://scikit-learn.org/stable/modules/svm.html#svm-regression
gridsearch: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
"""
"""
SVR w/ feature selection
"""
# import pandas as pd   # for import and export the data to csv
# from sklearn.svm import SVR
# from sklearn.metrics import mean_squared_error as mse
# from sklearn.metrics import make_scorer
# from sklearn.model_selection import cross_val_score
# import numpy as np
#
#
# def root_mean_squared_error(y_true, y_pred, sample_weight=None, multioutput='uniform_average'):
#     rmse = np.sqrt(mse(y_true, y_pred,
#                        sample_weight=sample_weight, multioutput=multioutput))
#     return rmse
#
# # import data
# # for training
# df_train = pd.read_csv('train-DR.csv')
# x_train_1 = df_train.drop(['Id','y','x6','x12','x13','x2'],axis=1).values
# y_train_1 = pd.read_csv('train-DR.csv').values[:,1]
#
# # for test
# df_test = pd.read_csv('test.csv')
# x_test_1= df_test.drop(['Id','x6','x12','x13','x2'],axis=1).values
# y_test = np.zeros([x_test_1.shape[0], 2])   # [id, y]
# y_test[:, 0] = df_test.values[:, 0]
#
# # support vector regression
# clf = SVR(kernel='rbf', C=1, gamma=0.01)
# clf.fit(x_train_1,y_train_1)
# # clf = SVR(kernel='linear', C=1e3)
# # clf = SVR(kernel='poly', C=1e3, degree=2, epsilon=0.05)
#
# cvScore = cross_val_score(clf,
#                           x_train_1, y_train_1,
#                           scoring=make_scorer(root_mean_squared_error),
#                           verbose=0,
#                           cv=10)
# print('Mean of CV RMSE:', np.mean(cvScore))
#
# y_test[:, 1] = clf.predict(x_test_1)
#
# df = pd.DataFrame(data=y_test)
# df.to_csv('./svr_rbf_cv=10.csv', header=['Id', 'y'], sep=',', index=False, float_format='%.5f')


"""
gridsearch
"""
import pandas as pd   # for import and export the data to csv
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import GridSearchCV

def root_mean_squared_error(y_true, y_pred, sample_weight=None, multioutput='uniform_average'):
    rmse = np.sqrt(mse(y_true, y_pred,
                       sample_weight=sample_weight, multioutput=multioutput))
    return rmse

# import data
# for training
rawDataTrain = pd.read_csv('train.csv').values
X_train = rawDataTrain[:, 2:]   # the features
y_train = rawDataTrain[:, 1]  # the target value

# tune the svr parameters
tuned_para = [{'C': [1, 10, 100,1000], 'gamma': [0.1, 0.01, 0.001, 0.0001], 'kernel': ['linear']},
              {'C': [1, 10, 100,1000], 'gamma': [0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}]
clf = GridSearchCV(estimator = SVR(), param_grid = tuned_para, cv=10, scoring=make_scorer(root_mean_squared_error,greater_is_better=False))
clf.fit(X_train, y_train)
print('best_parameter', clf.best_params_)

# for test
df_test = pd.read_csv('test.csv')
x_test = df_test.values[:,1:]
y_test = np.zeros([x_test.shape[0], 2])   # [id, y]
y_test[:, 0] = df_test.values[:, 0]

y_test[:, 1] = clf.predict(x_test)
df = pd.DataFrame(data=y_test)
df.to_csv('./svr_rbf_gridsearch.csv', header=['Id', 'y'], sep=',', index=False, float_format='%.5f')


"""
feature selection: http://scikit-learn.org/stable/modules/feature_selection.html
PCA:
"""
# import pandas as pd
# from sklearn.decomposition import PCA
# rawDataTrain = pd.read_csv('train.csv').values
# X_train = rawDataTrain[:, 2:]   # the features
# pca=PCA(n_components=15)
# pca.fit(X_train)
# print(pca.explained_variance_)
# print(pca.components_)
#
# # feature selection
# from sklearn.svm import LinearSVR
# from sklearn.feature_selection import SelectFromModel
# import pandas as pd
# import numpy as np
# rawDataTrain = pd.read_csv('train.csv').values
# X_train = rawDataTrain[:, 2:]   # the features
# Y_train = rawDataTrain[:, 1] # the y
# clf = LinearSVR(C=1).fit(X_train, Y_train)
# model = SelectFromModel(clf)
# X_new = model.transform((X_train))
# df = pd.DataFrame(data=X_new)
# df.to_csv('./X_new.csv', sep=',', index=False)
# print (X_new)


"""
NuSVR poly
"""
# import pandas as pd   # for import and export the data to csv
# from sklearn.metrics import mean_squared_error as mse
# from sklearn.metrics import make_scorer
# from sklearn.model_selection import cross_val_score
# import numpy as np
# from sklearn.svm import NuSVR
# from sklearn.linear_model import Lasso
#
# def root_mean_squared_error(y_true, y_pred, sample_weight=None, multioutput='uniform_average'):
#     rmse = np.sqrt(mse(y_true, y_pred,
#                        sample_weight=sample_weight, multioutput=multioutput))
#     return rmse
#
# # import data
# # for training
# rawDataTrain = pd.read_csv('train.csv').values
# X_train = rawDataTrain[:, 2:]   # the features
# y_train = rawDataTrain[:, 1]   # the target value
#
# # for test
# rawDataTest = pd.read_csv('test.csv').values
# X_test = rawDataTest[:, 1:]
# y_test = np.zeros([X_test.shape[0], 2])   # [id, y]
# y_test[:, 0] = rawDataTest[:, 0]
#
# # NuSVR
# clf = NuSVR(C=10, gamma=0.1,kernel='poly', degree=3)
# clf.fit(X_train, y_train)
#
# cvScore = cross_val_score(clf,
#                           X_train, y_train,
#                           scoring=make_scorer(root_mean_squared_error),
#                           verbose=0,
#                           cv=10)
# print('Mean of CV RMSE:', np.mean(cvScore))
#
# y_test[:, 1] = clf.predict(X_test)
# df = pd.DataFrame(data=y_test)
# df.to_csv('./nusvr.csv', header=['Id', 'y'], sep=',', index=False)