"""
SVR
"""
import pandas as pd   # for import and export the data to csv
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.svm import SVR

def root_mean_squared_error(y_true, y_pred, sample_weight=None, multioutput='uniform_average'):
    rmse = np.sqrt(mse(y_true, y_pred,
                       sample_weight=sample_weight, multioutput=multioutput))
    return rmse

# import data
# for training
rawDataTrain = pd.read_csv('train_statistical_moments.csv').values
X_train = rawDataTrain[:, 2:]   # the features
y_train = rawDataTrain[:, 1]   # the target value

# for test
rawDataTest = pd.read_csv('test_statistical_moments.csv').values
X_test = rawDataTest[:, 1:]
y_test = np.zeros([X_test.shape[0], 2])   # [id, y]
y_test[:, 0] = rawDataTest[:, 0]

# SVR
clf = SVR(C=1, gamma=0.01, kernel='poly')
clf.fit(X_train, y_train)

cvScore = cross_val_score(clf,
                          X_train, y_train,
                          scoring=make_scorer(root_mean_squared_error),
                          verbose=0,
                          cv=10)
print('Mean of CV RMSE:', np.mean(cvScore))

y_test[:, 1] = clf.predict(X_test)
df = pd.DataFrame(data=y_test)
df.to_csv('./svr.csv', header=['Id', 'y'], sep=',', index=False)

"""
NuSVR
"""
import pandas as pd   # for import and export the data to csv
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.svm import NuSVR

def root_mean_squared_error(y_true, y_pred, sample_weight=None, multioutput='uniform_average'):
    rmse = np.sqrt(mse(y_true, y_pred,
                       sample_weight=sample_weight, multioutput=multioutput))
    return rmse

# import data
# for training
rawDataTrain = pd.read_csv('train_statistical_moments.csv').values
X_train = rawDataTrain[:, 2:]   # the features
y_train = rawDataTrain[:, 1]   # the target value

# for test
rawDataTest = pd.read_csv('test_statistical_moments.csv').values
X_test = rawDataTest[:, 1:]
y_test = np.zeros([X_test.shape[0], 2])   # [id, y]
y_test[:, 0] = rawDataTest[:, 0]

# NuSVR
clf = NuSVR(C=10, gamma=0.1,kernel='poly',degree=3)
clf.fit(X_train, y_train)

cvScore = cross_val_score(clf,
                          X_train, y_train,
                          scoring=make_scorer(root_mean_squared_error),
                          verbose=0,
                          cv=10)
print('Mean of CV RMSE:', np.mean(cvScore))

y_test[:, 1] = clf.predict(X_test)
df = pd.DataFrame(data=y_test)
df.to_csv('./nusvr.csv', header=['Id', 'y'], sep=',', index=False)