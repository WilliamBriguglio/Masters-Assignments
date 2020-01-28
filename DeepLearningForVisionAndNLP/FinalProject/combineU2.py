import pandas as pd
import os
import numpy as np

total = 0
IDs_U2 = np.empty((1,1))
X_U2 = np.empty((1,183,183,6))
for i in range(0,30):
	try:
		IDs = pd.read_csv("U200934/B_"+str(i), header=None).to_numpy() 	#load batch file containing IDs of samples in batch i with size b/w 101400 and 200934
		x = np.load("U200934Samps/B_"+str(i)+".npz")['arr_0']	 		#load batch file containing input tensors of samples in batch i with size b/w 101400 and 200934
	
		X_U2 = np.append(X_U2, x, axis = 0)	#append batches input tensors
		IDs_U2 = np.append(IDs_U2, IDs, axis = 0)	#append batches IDs
		print(i)
	except pd.io.common.EmptyDataError:
		pass

#Save final tensor containing the entire dataset used by our keras model
np.savez_compressed("trainProc/X_U2.npz", X_U2[1:])
#Save the vector of IDs of the Samples in tehthe final dataset, with the IDs in the same order as the corresponding samples appear in the final tensor
np.savez_compressed("trainProc/ID_U2.npz", IDs_U2[1:])

print(X_U2.shape)
print(IDs_U2.shape)



