import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score as bac
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from joblib import dump

from treeinterpreter import treeinterpreter as ti
from sklearn.model_selection import train_test_split

#load data
X = pd.read_csv("X_train.csv", header=None, low_memory=False).to_numpy()
X_t = pd.read_csv("X_test.csv", header=None, low_memory=False).to_numpy()
Y = pd.read_csv("Y_train.csv", header=None, low_memory=False).to_numpy()
Y_t = pd.read_csv("Y_test.csv", header=None, low_memory=False).to_numpy()
features = pd.read_csv("feat_Names.csv", header=None, low_memory=False).to_numpy()

print("\nMalicious Test Samples(Class = 1):", int(np.sum(Y)),"   Benign Test Samples(Class = 0):", Y.shape[0] - int(np.sum(Y)))
print("Malicious Train Samples(Class = 1):", int(np.sum(Y_t)),"   Benign Train Samples(Class = 0):", Y_t.shape[0] - int(np.sum(Y_t)),"\n" )


#Decision Tree cross validated grid search and training
DTC = DecisionTreeClassifier(class_weight='balanced')
params = {'criterion':('gini', 'entropy'), 'min_samples_leaf':[0.00000001,0.0000001,0.000001,0.00001,0.0001,0.001,0.01] }
GS = GridSearchCV(DTC, params, cv=5, scoring='balanced_accuracy', n_jobs=-1)
GS.fit(X, Y)
print("The best score was ",round((GS.best_score_*100),4)," using the params:\n",GS.best_params_)
DTC = GS.best_estimator_
dump(DTC, 'DTC.joblib')


#Linear Regression cross validated grid search and training
LR = LogisticRegression(class_weight='balanced', max_iter=10000, solver='liblinear')
params = {'tol':[0.0001,0.00001,0.000001,0.0000001], 'C':[0.1,1,10,100] }
GS = GridSearchCV(LR, params, cv=5, scoring='balanced_accuracy', n_jobs=-1)
GS.fit(X, Y.ravel())
print("The best score was ",round((GS.best_score_*100),4)," using the params:\n",GS.best_params_)
LR = GS.best_estimator_
dump(LR, 'LR.joblib')

#k-Nearest Neighbour cross validated grid search and training
kNN = KNeighborsClassifier()
#params = {'n_neighbors':[3,5,10,15,20], 'weights':['uniform','distance'] }
params = {'n_neighbors':[3], 'weights':['distance'] }
GS = GridSearchCV(kNN, params, cv=5, scoring='balanced_accuracy', n_jobs=-1)
GS.fit(X, Y.ravel())
print("The best score was ",round((GS.best_score_*100),4)," using the params:\n",GS.best_params_)
kNN = GS.best_estimator_
Y_pred = kNN.predict(X_t)
print("kNN balanced accuracy before Feat Sel: ", bac(Y_t, Y_pred)) #print validation results to compare with k-NN after feature selection
dump(kNN, 'kNN.joblib')

#k-Nearest Neighbour cross validated grid search and training after Feature selection 
model = SelectFromModel(DTC, prefit=True)
X_new = model.transform(X)
X_t_new = model.transform(X_t)
sel_Ind = model.get_support(indices=True)
print(X.shape)
print(X_new.shape)
kNN = KNeighborsClassifier()
kNN =  kNN.fit(X_new, Y.ravel())
Y_pred = kNN.predict(X_t_new)
print("kNN balanced accuracy after Feat Sel: ", bac(Y_t, Y_pred)) #print validation results to compare with k-NN before feature selection
np.savetxt("X_train_reduced.csv", X_new, delimiter=",")
np.savetxt("X_test_reduced.csv", X_t_new, delimiter=",")
fp = open("SelectedIndices.csv","w")
for i in range(len(sel_Ind)-1):
    fp.write(str(sel_Ind[i])+",")
fp.write(str(sel_Ind[i+1])+"\n")
fp.close()
dump(kNN, 'kNNReduced.joblib')
