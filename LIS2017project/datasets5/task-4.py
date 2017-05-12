"""
MLP-train labeled data
"""
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.optimizers import SGD
from keras import regularizers
import numpy as np
import pandas as pd

train_labeled=pd.read_hdf('train_labeled.h5','train').values
train_unlabeled=pd.read_hdf('train_unlabeled.h5','train').values
test_data=pd.read_hdf('test.h5','test').values

X_trainlab=train_labeled[:,1:]
y_trainlab=train_labeled[:,0]
X_trainun=train_unlabeled[:,:]
X_test=test_data[:,:]
y_test=np.zeros([X_test.shape[0],2])
y_test[:,0]=pd.read_csv('sample.csv', index_col=False).values[:,0]

from keras.utils.np_utils import to_categorical
y_trainlab=to_categorical(y_trainlab)

model=Sequential()
model.add(Dense(256,activation='relu',input_dim=128))
model.add(Dropout(0.5))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['categorical_accuracy'])

model.fit(X_trainlab,y_trainlab,epochs=500,batch_size=128)
score=model.evaluate(X_trainlab,y_trainlab,batch_size=128)
print(score)

y_test[:,1]=model.predict_classes(X_test)
df=pd.DataFrame(data=y_test)
df.to_csv('MLP-trainlabeled.csv', header=['Id', 'y'], sep=',', index=False, float_format='%.0f')

"""
cluster-unlabel-low accuracy
"""
# from sklearn.cluster import KMeans
# import sklearn.metrics as sm
# import pandas as pd
# import numpy as np
#
#
# # train_labeled=pd.read_hdf('train_labeled.h5','train').values
# train_unlabeled=pd.read_hdf('train_unlabeled.h5','train').values
# test_data=pd.read_hdf('test.h5','test').values
#
# # X_trainlab=train_labeled[:,1:]
# # y_trainlab=train_labeled[:,0]
# X_trainun=train_unlabeled[:,:]
# X_test=test_data[:,:]
# y_test=np.zeros([X_test.shape[0],2])
# y_test[:,0]=pd.read_csv('sample.csv', index_col=False).values[:,0]
#
# model=KMeans(n_clusters=10,max_iter=1000)
# model.fit(X_trainun)
#
# model.labels_
# print(model.labels_)
#
# y_test=np.zeros([X_test.shape[0],2])
# y_test[:,0]=pd.read_csv('sample.csv', index_col=False).values[:,0]
# y_test[:,1]=model.predict(X_test)
# df=pd.DataFrame(data=y_test)
# df.to_csv('Cluster-trainunlabeled.csv', header=['Id', 'y'], sep=',', index=False, float_format='%.0f')
