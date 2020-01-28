import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV

from treeinterpreter import treeinterpreter as ti
from sklearn.model_selection import train_test_split

#load data
X = pd.read_csv("X_test.csv", header=None, low_memory=False).to_numpy()
Y = pd.read_csv("Y_test.csv", header=None, low_memory=False).to_numpy()

print("\nMalicious Samples(Class = 1):", int(np.sum(Y)),"\tBenign Samples(Class = 0)", Y.shape[0] - int(np.sum(Y)),"\n" )

#save split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
np.savetxt("X2_train.csv", X_train, delimiter=",")
np.savetxt("X2_test.csv", X_test, delimiter=",")
np.savetxt("Y2_train.csv", Y_train, delimiter=",")
np.savetxt("Y2_test.csv", Y_test, delimiter=",")

print("Train and test split for ensemlble classifier saved")
