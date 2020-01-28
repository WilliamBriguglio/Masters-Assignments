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

#Load data
X_test = pd.read_csv("X_test.csv", header=None, low_memory=False).to_numpy()
Y_test = pd.read_csv("Y_test.csv", header=None, low_memory=False).to_numpy()
X2_test = pd.read_csv("X2_test.csv", header=None, low_memory=False).to_numpy()
Y2_test = pd.read_csv("Y2_test.csv", header=None, low_memory=False).to_numpy()

print("\nMalicious Trian Samples in set 1(Class = 1):", int(np.sum(Y_test)),"   Benign Test Samples in set 1(Class = 0):", Y_test.shape[0] - int(np.sum(Y_test)))
print("Malicious Trian Samples in set 2(Class = 1):", int(np.sum(Y2_test)),"   Benign Test Samples in set 1(Class = 0):", Y2_test.shape[0] - int(np.sum(Y2_test)))

#Load models
DTC = load('DTC.joblib')
LR = load('LR.joblib')
kNN = load('kNN.joblib')
DTC_ensemble = load('DTC_ensemble.joblib')
LR_ensemble = load('LR_ensemble.joblib')
kNN_ensemble = load('kNN_ensemble.joblib')
DTC_ensemble2 = load('DTC_ensemble2.joblib')
LR_ensemble2 = load('LR_ensemble2.joblib')
kNN_ensemble2 = load('kNN_ensemble2.joblib')

#obtain predictions for single models
Y_DTC = DTC.predict(X_test)
Y_LR = LR.predict(X_test)
Y_kNN = kNN.predict(X_test)

#obtain features for scheme 1
DTC_pred = DTC.predict(X_test)
LR_pred = LR.predict(X_test)
kNN_pred = kNN.predict(X_test)
X_test_pred = np.stack((DTC_pred, LR_pred, kNN_pred), axis = -1)

#obtain predictions for ensemble models using schceme 1
Y_DTC_ensemble = DTC_ensemble.predict(X_test_pred)
Y_LR_ensemble = LR_ensemble.predict(X_test_pred)
Y_kNN_ensemble = kNN_ensemble.predict(X_test_pred)

#obtain features for schceme 2
DTC_pred = DTC.predict(X2_test)
LR_pred = LR.predict(X2_test)
kNN_pred = kNN.predict(X2_test)
X_test_pred = np.stack((DTC_pred, LR_pred, kNN_pred), axis = -1)


#obtain predictions for ensemble models using schceme 2
Y_DTC_ensemble2 = DTC_ensemble2.predict(X_test_pred)
Y_LR_ensemble2 = LR_ensemble2.predict(X_test_pred)
Y_kNN_ensemble2 = kNN_ensemble2.predict(X_test_pred)

#calculate and prind balanced accuracy scores
print("\nBALANCED ACCURACY SCORES:")
print("------------------------------------------------------------")
print("DTC: ",bac(Y_test, Y_DTC))
print("LR: ",bac(Y_test, Y_LR))
print("kNN: ",bac(Y_test, Y_kNN))
print("DTC_ensemble: ",bac(Y_test, Y_DTC_ensemble))
print("LR_ensemble: ",bac(Y_test, Y_LR_ensemble))
print("kNN_ensemble: ",bac(Y_test, Y_kNN_ensemble))
print("DTC_ensemble2: ",bac(Y2_test, Y_DTC_ensemble2))
print("LR_ensemble2: ",bac(Y2_test, Y_LR_ensemble2))
print("kNN_ensemble2: ",bac(Y2_test, Y_kNN_ensemble2))
