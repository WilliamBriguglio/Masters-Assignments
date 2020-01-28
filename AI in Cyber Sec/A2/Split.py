import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV

from treeinterpreter import treeinterpreter as ti
from sklearn.model_selection import train_test_split

#load data
df = pd.read_csv("dataSet/dataset_malwares.csv", header=None, low_memory=False).to_numpy()


#extract features names, labels, and samples
features = df[0,1:]
features = np.delete(features, 52)
Y = df[1:,53].astype(int)
X = df[1:,1:].astype(float)
X = np.delete(X, 52, 1)

print("\nMalicious Samples(Class = 1):", int(np.sum(Y)),"\tBenign Samples(Class = 0)", Y.shape[0] - int(np.sum(Y)),"\n" )

#Save split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10)
np.savetxt("X_train.csv", X_train, delimiter=",")
np.savetxt("X_test.csv", X_test, delimiter=",")
np.savetxt("Y_train.csv", Y_train, delimiter=",")
np.savetxt("Y_test.csv", Y_test, delimiter=",")
fp = open("feat_Names.csv", "w+")
for i in range(len(features)-1):
    fp.write(features[i]+",")
fp.write(features[i+1]+"\n")

print("Train and test split saved")
