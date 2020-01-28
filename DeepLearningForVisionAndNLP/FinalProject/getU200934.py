
#__________________IMPORTS_________________________________________________________
import sys
import os
import re
import numpy as np
import pandas as pd

#__________________MAIN_________________________________________________________

#read batch file which contains IDs of files win legnth in the range 101400 to 200934
files = pd.read_csv("batches/"+sys.argv[1], header = None).to_numpy()

U200934 = []

# for each ID
for i in files:

	#load bytes file
	fp = open("train/"+i[0],"r")
	bytes = fp.read()
	fp.close()

	#clean data
	bytes = re.sub(r'([0-9]|[A-F]){8}',"",bytes)
	bytes = re.sub(r'\n'," ",bytes)
	bytes = re.sub(r'  '," ",bytes)

	#split into list of byte pixels
	pixels = [token for token in bytes.split(" ") if token != ""]
	
	#remove anomalous
	l = len(pixels)
	for t in range(l):
		if pixels[t] == "??":
			pixels[t] = "00"
		pixels[t] = int(pixels[t], 16)

	#select only files in range 101,400 and 200,934
	if l <= 200934 and l >= 200934:
		U200934.append(i[0])

fp = open("U200934/"+sys.argv[1], "w")
for i in U200934:
	fp.write(i+"\n")
fp.close()

print("\n\n\n\t\t\t-> DONE! <-\n\n")





