import pandas as pd
import numpy as np

#load tensor containing IDs if final data set
IDs = np.load("trainProc/ID_U2.npz", allow_pickle=True)['arr_0']
#load array containg IDs and their correspnding label
labels = pd.read_csv("trainLabels.csv").to_numpy()

#create vector of labels to use for classification
Y = []
for i in IDs:	
	indx = np.where(labels[:,0] == i[0][:20])[0][0]
	y = labels[indx,1]	
	Y.append(y)
Y= np.array(Y)

print(Y.shape)
np.save("trainProc/Y_U2.npy", Y)