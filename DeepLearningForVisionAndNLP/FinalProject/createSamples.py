import numpy as np 
import re
import pandas as pd
import sys

try: #read batch file which contains IDs of files win legnth in the range 101400 to 200934
	files = pd.read_csv(sys.argv[1], header=None).to_numpy()
except pd.io.common.EmptyDataError:
	exit()

U200934 = np.empty((1,183,183,6))

#for each ID
for i in files:

	#load bytes file for sample with ID i
	fp = open("train/"+i[0], "r")
	bytes = fp.read()
	fp.close()

	#clean data
	bytes = re.sub(r'([0-9]|[A-F]){8}',"",bytes)
	bytes = re.sub(r'\n'," ",bytes)
	bytes = re.sub(r'  '," ",bytes)
	bytes = re.sub(r'\?\?',"00", bytes)
	
	#convert to array of hex pixels
	data = [int(token, 16) for token in bytes.split(" ") if token != ""]
	
	l = len(data)

	if l <= 200934 and l > 101400:
	
		#pad so last row has length 183
		smallpad2 = (183*6) - (l % (183*6))
		if smallpad2 == (183*6):
			smallpad2 =0
		pixels = data + ([0] * smallpad2)
		pixels = np.reshape(pixels, (-1,183,6))
		
		#padd equally on top and bottom so shape becomes (183, 183, 6)
		dif = 183 - pixels.shape[0]
		frontpad = np.zeros((int(dif/2),183,6))
		if (dif % 2) == 0:
			backpad = np.zeros((int(dif/2),183,6))
		else:
			backpad = np.zeros((int(dif/2)+1,183,6))
		pixels = np.concatenate((frontpad, pixels, backpad))
		
		#add batch axis
		pixels = np.expand_dims(pixels, axis=0)
		#append sample to batch tensor
		U200934 = np.append(U200934, pixels, axis=0)


print(U200934.shape)
#save batch tensor to U200934Samps
np.savez_compressed('U200934Samps/'+sys.argv[1][8:]+'.npz', U200934[1:])
