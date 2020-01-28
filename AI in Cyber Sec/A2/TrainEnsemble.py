import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score as bac
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from joblib import load, dump

from treeinterpreter import treeinterpreter as ti
from sklearn.model_selection import train_test_split


print("------------------------SCHEME 1----------------------------")
#Load data
X = pd.read_csv("X_train.csv", header=None, low_memory=False).to_numpy()
Y = pd.read_csv("Y_train.csv", header=None, low_memory=False).to_numpy()

print("\nMalicious Trian Samples(Class = 1):", int(np.sum(Y)),"   Benign Test Samples(Class = 0):", Y.shape[0] - int(np.sum(Y)))

#Load models
DTC = load('DTC.joblib')
LR = load('LR.joblib')
kNN = load('kNN.joblib')

#create features for ensemble models
DTC_pred = DTC.predict(X)
LR_pred = LR.predict(X)
kNN_pred = kNN.predict(X)

#final feature array
X_pred = np.stack((DTC_pred, LR_pred, kNN_pred), axis = -1)
print(X_pred.shape)

#Decision Tree cross validated grid search and training
DTC = DecisionTreeClassifier(class_weight='balanced')
params = {'criterion':('gini', 'entropy'), 'min_samples_leaf':[0.00000001,0.0000001,0.000001,0.00001,0.0001,0.001,0.01] }
GS = GridSearchCV(DTC, params, cv=5, scoring='balanced_accuracy', n_jobs=-1)
GS.fit(X_pred, Y)
print("The best score was ",round((GS.best_score_*100),4)," using the params:\n",GS.best_params_)
DTC = GS.best_estimator_
dump(DTC, 'DTC_ensemble.joblib')


#Linear Regression cross validated grid search and training
LR = LogisticRegression(class_weight='balanced', max_iter=10000, solver='liblinear')
params = {'tol':[0.0001,0.00001,0.000001,0.0000001], 'C':[0.1,1,10,100,1000] }
GS = GridSearchCV(LR, params, cv=5, scoring='balanced_accuracy', n_jobs=-1)
GS.fit(X_pred, Y.ravel())
print("The best score was ",round((GS.best_score_*100),4)," using the params:\n",GS.best_params_)
LR = GS.best_estimator_
dump(LR, 'LR_ensemble.joblib')


#Linear Regression cross validated grid search and training
kNN = KNeighborsClassifier()
params = {'n_neighbors':[3,5,10,15,20], 'weights':['uniform','distance'] }
GS = GridSearchCV(kNN, params, cv=5, scoring='balanced_accuracy', n_jobs=-1)
GS.fit(X_pred, Y.ravel())
print("The best score was ",round((GS.best_score_*100),4)," using the params:\n",GS.best_params_)
kNN = GS.best_estimator_
dump(kNN, 'kNN_ensemble.joblib')


print("------------------------SCHEME 2----------------------------")
#Load data
X = pd.read_csv("X2_train.csv", header=None, low_memory=False).to_numpy()
Y = pd.read_csv("Y2_train.csv", header=None, low_memory=False).to_numpy()

print("\nMalicious Test Samples(Class = 1):", int(np.sum(Y)),"   Benign Test Samples(Class = 0):", Y.shape[0] - int(np.sum(Y)))

#Load models
DTC = load('DTC.joblib')
LR = load('LR.joblib')
kNN = load('kNN.joblib')

#create features for ensemble models
DTC_pred = DTC.predict(X)
LR_pred = LR.predict(X)
kNN_pred = kNN.predict(X)

#final feature array
X_pred = np.stack((DTC_pred, LR_pred, kNN_pred), axis = -1)
print(X_pred.shape)

#Decision Tree cross validated grid search and training
DTC = DecisionTreeClassifier(class_weight='balanced')
params = {'criterion':('gini', 'entropy'), 'min_samples_leaf':[0.00000001,0.0000001,0.000001,0.00001,0.0001,0.001,0.01] }
GS = GridSearchCV(DTC, params, cv=5, scoring='balanced_accuracy', n_jobs=-1)
GS.fit(X_pred, Y)
print("The best score was ",round((GS.best_score_*100),4)," using the params:\n",GS.best_params_)
DTC = GS.best_estimator_
dump(DTC, 'DTC_ensemble2.joblib')


#Linear Regression cross validated grid search and training
LR = LogisticRegression(class_weight='balanced', max_iter=10000, solver='liblinear')
params = {'tol':[0.0001,0.00001,0.000001,0.0000001], 'C':[0.1,1,10,100,1000,10000,100000] }
GS = GridSearchCV(LR, params, cv=5, scoring='balanced_accuracy', n_jobs=-1)
GS.fit(X_pred, Y.ravel())
print("The best score was ",round((GS.best_score_*100),4)," using the params:\n",GS.best_params_)
LR = GS.best_estimator_
dump(LR, 'LR_ensemble2.joblib')


#Linear Regression cross validated grid search and training
kNN = KNeighborsClassifier()
params = {'n_neighbors':[1,2,3,5,10,15,20], 'weights':['uniform','distance'] }
GS = GridSearchCV(kNN, params, cv=5, scoring='balanced_accuracy', n_jobs=-1)
GS.fit(X_pred, Y.ravel())
print("The best score was ",round((GS.best_score_*100),4)," using the params:\n",GS.best_params_)
kNN = GS.best_estimator_
dump(kNN, 'kNN_ensemble2.joblib')
