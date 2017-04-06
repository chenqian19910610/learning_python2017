"""
multi-class prediction evaluation:http://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
http://scikit-learn.org/stable/modules/multiclass.html
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import NuSVC
from sklearn.svm import SVC

def accu_score(y_true,y_pred):
    acc=accuracy_score(y_true,y_pred)
    return acc

Train_data=pd.read_csv('train.csv').values
X_train=Train_data[:,2:]
y_train=Train_data[:,1]

Test_data=pd.read_csv('test.csv').values
X_test=Test_data[:,1:]
y_test=np.zeros([X_test.shape[0],2])
y_test[:,0]=Test_data[:,0]

clf=OneVsRestClassifier(SVC(kernel='poly',C=10, gamma=0.1))
clf.fit(X_train,y_train)
# y_pred=clf.predict(X_train)
# acc=accuracy_score(y_train,y_pred)
# print('accuracy',acc)

cvscore=cross_val_score(clf,X_train,y_train,scoring=make_scorer(accu_score),verbose=0,cv=10)
print('cvscore',np.mean(cvscore))

y_test[:,1]=clf.predict(X_test)
pd.DataFrame(data=y_test).to_csv('./onevsrest.csv', header=['Id', 'y'], sep=',', index=False)