
#__________________IMPORTS_________________________________________________________
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold as SKF
from sklearn.preprocessing import LabelEncoder



#__________________MAIN_________________________________________________________

#read in sample ID's
data = pd.read_csv("trainLabels.csv").to_numpy()
IDs = data[:,0]

k = 0
batches = np.array_split(IDs, 30) #split IDs into 30 batches

#for each batch, create file that is list of IDS within the batch and save the file to the folder "batches"
for b in batches:	
	fp = open("batches/B_" + str(k),'w')
	for ID in b:
		fp.write(ID+".bytes\n")
	k+=1
