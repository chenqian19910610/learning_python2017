"""
merge a new file, then do label propagation
semi-supervised learning
"""
# from keras.models import Sequential
# from keras.layers import Dense,Dropout,Activation
# from keras.optimizers import SGD
# from keras import regularizers
# import numpy as np
# import pandas as pd
# from sklearn.semi_supervised import label_propagation
#
# train_labeled=pd.read_hdf('train_labeled.h5','train').values
# train_unlabeled=pd.read_hdf('train_unlabeled.h5','train').values
# test_data=pd.read_hdf('test.h5','test').values
#
# train=np.zeros([train_labeled.shape[0]+train_unlabeled.shape[0], train_labeled.shape[1]])
# train[0:train_labeled.shape[0],0]=train_labeled[:,0]
# train[train_labeled.shape[0]:,0]=-1
# train[0:train_labeled.shape[0],1:]=train_labeled[:,1:]
# train[train_labeled.shape[0]:,1:]=train_unlabeled[:,:]
# print(train.shape)
#
# X_train=train[:,1:]
# y_train=train[:,0]
#
# lp_model=label_propagation.LabelSpreading()
# lp_model.fit(X_train,y_train)
# predict_label=lp_model.transduction_[train_labeled.shape[0]:]
# train[train_labeled.shape[0]:,0]=predict_label[:]
# y_train=train[:,0]
# X_test=test_data[:,:]
# y_test=np.zeros([X_test.shape[0],2])
# y_test[:,0]=pd.read_csv('sample.csv', index_col=False).values[:,0]
#
#
# from keras.utils.np_utils import to_categorical
# y_trainlab=to_categorical(y_train)
#
# model=Sequential()
# model.add(Dense(256,activation='relu',input_dim=128))
# model.add(Dropout(0.5))
# model.add(Dense(256,activation='relu'))
# model.add(Dropout(0.25))
# model.add(Dense(128,activation='relu'))
# model.add(Dropout(0.25))
# model.add(Dense(64,activation='relu'))
# model.add(Dropout(0.25))
# model.add(Dense(10,activation='softmax'))
# model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['categorical_accuracy'])
#
# model.fit(X_train,y_train,epochs=500,batch_size=300)
# score=model.evaluate(X_train,y_train,batch_size=300)
# print(score)
#
# y_test[:,1]=model.predict_classes(X_test)
# df=pd.DataFrame(data=y_test)
# df.to_csv('semisupervise_label prop.csv', header=['Id', 'y'], sep=',', index=False, float_format='%.0f')

"""
label propogation & nn
"""
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.optimizers import SGD
from keras import regularizers
import numpy as np
import pandas as pd

# label the unlabeled data using labeled data K-NN
class Labelpropagation:
    def __init__(self, n_nearest_neigh=10,iters=10,normalize=True, verbose=False): # use double underscore to construct the direct call function
        self.n_nearest_neigh=n_nearest_neigh
        self.neigh_indices=None
        self.iters=iters
        self.normalize=normalize
        self.verbose=verbose

    def _precompute_near_neigh_indices(self,X):
        self._neigh_indices=np.zeros((X.shape[0],self.n_nearest_neigh),dtype=int)
        if self.verbose:
            print('filling neighbor row 1/{}'.format(X.shape[0]))
        for i in range(len(X)):
            if self.verbose and (i%500==0 and i>1):
                print(i,end='')
            dists=np.sum((X-X[i])**2,axis=1)
            nearest_indices=np.argpartition(dists,self.n_nearest_neigh)[:(self.n_nearest_neigh+1)]
            nearest_indices=np.array([j for j in nearest_indices if j!=i])
            self._neigh_indices[i]=nearest_indices
        pass

    def _avg_nearby_labels(self,ytr,nearest_indices):
        nearest_labels=ytr[nearest_indices]
        nearest_labels=nearest_labels.sum(axis=0)
        nearest_labels/=nearest_labels.sum()
        return nearest_labels

    def _propagate_labels(self,ytr):
        if len(ytr.shape)!=2:
            raise ValueError('ytr has shape {},must be one-hot(2D)'.format(ytr.shape))
        n=self._neigh_indices.shape[0]-ytr.shape[0]
        yunl_proba=np.ones((n,ytr.shape[1]))/ytr.shape[1]
        new_labels=np.zeros_like(yunl_proba)
        for j in range(self.iters):
            ys=np.vstack((ytr,yunl_proba))
            for i in range(n):
                new_labels[i]=self._avg_nearby_labels(ys,self._neigh_indices[ytr.shape[0]+i])
            yunl_proba=new_labels.copy()
        if self.normalize:
            yunl_proba/yunl_proba
        yunl=yunl_proba.argmax(axis=1)
        return yunl

    def propagate(self,Xtr,ytr,Xunl):
        self._precompute_near_neigh_indices(np.vstack((Xtr,Xunl)))
        yunl = self._propagate_labels((ytr))
        return yunl

print("start")
Xtr=pd.read_hdf('train_labeled.h5','train').values[:,1:]
ytr=pd.read_hdf('train_labeled.h5','train').values[:,0]
nb_labels=10
ytr=np.eye(nb_labels)[np.array(ytr).astype(np.int64)] # convert y array into onehot(2D)
Xunl=pd.read_hdf('train_unlabeled.h5','train').values[:,:]
labelprop = Labelpropagation() #given the absolute value, no need to write parameter
y=labelprop.propagate(Xtr,ytr,Xunl)
print(y)
print("end")

#NN train combined data
train=np.zeros([Xtr.shape[0]+Xunl.shape[0], Xtr.shape[1]+1])
train[0:Xtr.shape[0],0]=pd.read_hdf('train_labeled.h5','train').values[:,0]
train[Xtr.shape[0]:,0]=y[:]
train[0:Xtr.shape[0],1:]=Xtr[:,:]
train[Xtr.shape[0]:,1:]=Xunl[:,:]
print(train)

X_train=train[:,1:]
y_train=train[:,0]

from keras.utils.np_utils import to_categorical
y_train=to_categorical(y_train)

model=Sequential()
model.add(Dense(256,activation='relu',input_dim=128))
model.add(Dropout(0.5))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['categorical_accuracy'])

model.fit(X_train,y_train,epochs=500,batch_size=300)
score=model.evaluate(X_train,y_train,batch_size=300)
print(score)

X_test=pd.read_hdf('test.h5','test').values[:,:]
y_test=np.zeros([X_test.shape[0],2])
y_test[:,0]=pd.read_csv('sample.csv', index_col=False).values[:,0]
y_test[:,1]=model.predict_classes(X_test)
df=pd.DataFrame(data=y_test)
df.to_csv('semisupervise_label prop_2.csv', header=['Id', 'y'], sep=',', index=False, float_format='%.0f')