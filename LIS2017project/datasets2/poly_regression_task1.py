"""
Packages: sklearn: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model
          pandas: http://pandas.pydata.org/pandas-docs/stable/io.html#io-store-in-csv
LassoCV: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html#examples-using-sklearn-preprocessing-polynomialfeatures
Polinomial features:http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html#examples-using-sklearn-preprocessing-polynomialfeatures
"""
# import pandas as pd   # for import and export the data to csv
# from sklearn.metrics import mean_squared_error as mse
# from sklearn.metrics import make_scorer
# from sklearn.model_selection import cross_val_score
# import numpy as np
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LassoCV
#
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
# X_train = PolynomialFeatures(degree=3).fit_transform(X_train)
#
#
# # for test
# rawDataTest = pd.read_csv('test.csv').values
# X_test = rawDataTest[:, 1:]
# X_test = PolynomialFeatures(degree=3).fit_transform(X_test)
# y_test = np.zeros([X_test.shape[0], 2])   # [id, y]
# y_test[:, 0] = rawDataTest[:, 0]
#
# # Lassocv
# clf = LassoCV(cv=10)
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
# df.to_csv('./lassocv.csv', header=['Id', 'y'], sep=',', index=False)

"""
visualize PCA-2Dimension
"""
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from sklearn.decomposition import PCA
# from sklearn.svm import SVR
#
# rawDataTrain = pd.read_csv('train.csv').values
# X_train = rawDataTrain[:, 2:]
# y_train = rawDataTrain[:, 1]
# X_train_1 = PCA(n_components=2).fit(X_train).transform(X_train)
# print(X_train_1.shape)
#
# clf=SVR(kernel='rbf',C=1)
# clf.fit(X_train_1,y_train)
#
# h=0.02
# x1_min,x1_max=X_train_1[:,0].min()-1,X_train_1[:,0].max()+1
# x2_min,x2_max=X_train_1[:,1].min()-1,X_train_1[:,1].max()+1
# xx,yy=np.meshgrid(np.arange(x1_min,x1_max,h),np.arange(x2_min,x2_max,h))
# Z=clf.predict(np.c_[xx.ravel(),yy.ravel()])
# Z=Z.reshape(xx.shape)
#
# plt.contourf(xx,yy,Z,alpha=0.3)
# plt.title('PCA-SVC')
# plt.xlim(x1_min,x1_max)
# plt.ylim(x2_min,x2_max)
# plt.xlabel('x1')
# plt.ylabel('x2')
# plt.scatter(X_train_1[:,0],X_train_1[:,1],c=y_train,alpha=0.8)
#
# plt.show()
