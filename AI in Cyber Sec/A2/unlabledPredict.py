import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from joblib import load



#Load data
df = pd.read_csv("dataSet/dataset_test.csv", low_memory=False).to_numpy()
X = df[:,1:]
labels = df[:,0]

#make sinlge predictions
DTC = load('DTC.joblib')
LR = load('LR.joblib')
kNN = load('kNN.joblib')

LR_ensemble2 = load("kNN_ensemble2.joblib")

#create features
DTC_pred = DTC.predict(X)
LR_pred = LR.predict(X)
kNN_pred = kNN.predict(X)
X_pred = np.stack((DTC_pred, LR_pred, kNN_pred), axis = -1)

#make ensemble prediction
Y_pred = LR_ensemble2.predict(X_pred)

#save predictions
print("Unlabeled Data predictions:")
for i in range(len(Y_pred)):
    print("Sample Name:",labels[i])
    print("\tprediction:",Y_pred[i],"\n")
