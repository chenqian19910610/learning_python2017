# import numpy as np
# import pandas as pd
# from sklearn import svm, datasets
# import matplotlib.pyplot as plt
#
# iris=datasets.load_iris()
# X=iris.data[:,:2]
# y=iris.target
#
# h=0.02 #stepsize in the mesh
#
# clf=svm.SVC(kernel='rbf',C=1)
# clf.fit(X,y)
#
# X_min,X_max=X[:,0].min()-1,X[:,0].max()+1
# y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
# xx,yy=np.meshgrid(np.arange(X_min,X_max,h),np.arange(y_min,y_max,h))  #xx,yy is matrix/tuple, requires ravel to two lines.
#
# titles=['SVC with linear kernel']
#
# Z=clf.predict(np.c_[xx.ravel(),yy.ravel()]) #only values can be plotted, xx,yy cannot be plotted
# Z=Z.reshape(xx.shape)

# plt.contourf(xx,yy,Z,alpha=0.5)
# plt.scatter(X[:,0],X[:,1],c=y,alpha=0.8)
# plt.xlabel('x1')
# plt.ylabel('x2')
# plt.xlim(xx.min(),xx.max())
# plt.ylim(yy.min(),yy.max())
# plt.xticks(())
# plt.yticks(())
# plt.title(titles)
#
# plt.show()

