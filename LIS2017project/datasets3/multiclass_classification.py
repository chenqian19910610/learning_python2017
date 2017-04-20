"""
multi-class prediction evaluation:http://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
http://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification
http://scikit-learn.org/stable/modules/multiclass.html
"""
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import cross_val_score
# from sklearn.metrics import make_scorer
# from sklearn.metrics import accuracy_score
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.multiclass import OneVsOneClassifier
# from sklearn.svm import NuSVC
# from sklearn.svm import SVC
#
#
# def accu_score(y_true,y_pred):
#     acc=accuracy_score(y_true,y_pred)
#     return acc
#
# Train_data=pd.read_csv('train.csv').values
# X_train=Train_data[:,2:]
# y_train=Train_data[:,1]
#
# Test_data=pd.read_csv('test.csv').values
# X_test=Test_data[:,1:]
# y_test=np.zeros([X_test.shape[0],2])
# y_test[:,0]=Test_data[:,0]
#
# clf=OneVsRestClassifier(NuSVC(nu=0.3, kernel='poly', degree=3, gamma=0.1))
# clf.fit(X_train,y_train)
# # y_pred=clf.predict(X_train)
# # acc=accuracy_score(y_train,y_pred)
# # print('accuracy',acc)
#
# cvscore=cross_val_score(clf,X_train,y_train,scoring=make_scorer(accu_score),verbose=0,cv=10)
# print('cvscore',np.mean(cvscore))
#
# y_test[:,1]=clf.predict(X_test)
# pd.DataFrame(data=y_test).to_csv('./onevsrest.csv', header=['Id', 'y'], sep=',', index=False)


"""
neural network classifier, KNeighbor, Decision Tree,SGDClass
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import PolynomialFeatures


def accu_score(y_true,y_pred):
    acc=accuracy_score(y_true,y_pred)
    return acc

rawDataTrain = pd.read_csv('train.csv').values
X_train = rawDataTrain[:, 2:]   # the features
y_train = rawDataTrain[:, 1]   # the target value
X_train = PolynomialFeatures(degree=3).fit_transform(X_train)

rawDataTest = pd.read_csv('test.csv').values
X_test = rawDataTest[:, 1:]
X_test = PolynomialFeatures(degree=3).fit_transform(X_test)
y_test = np.zeros([X_test.shape[0], 2])   # [id, y]
y_test[:, 0] = rawDataTest[:, 0]

# clf=DecisionTreeClassifier(random_state=0)
# clf=KNeighborsClassifier(n_neighbors=3)
# clf=SGDClassifier(loss='hinge')
# clf=MLPClassifier(solver='lbfgs', alpha=1e-5)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=300, criterion='entropy', max_features=None,
                                   bootstrap=True, min_samples_leaf=1, max_leaf_nodes=None, class_weight='balanced_subsample')
clf.fit(X_train,y_train)
# y_pred=clf.predict(X_train)
# acc=accuracy_score(y_train,y_pred)
# print('accuracy',acc)

cvscore=cross_val_score(clf,X_train,y_train,scoring=make_scorer(accu_score),verbose=0,cv=10)
print('cvscore',np.mean(cvscore))

y_test[:,1]=clf.predict(X_test)
pd.DataFrame(data=y_test).to_csv('./decisiontree.csv', header=['Id', 'y'], sep=',', index=False)