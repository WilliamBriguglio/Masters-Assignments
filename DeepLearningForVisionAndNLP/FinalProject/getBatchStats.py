
#__________________IMPORTS_________________________________________________________
import sys
import os
import re
import numpy as np
import pandas as pd


#__________________MAIN_________________________________________________________

#read batch file which contains IDs of files
files = pd.read_csv("batches/"+sys.argv[1], header = None).to_numpy()

#keep track of counts of each binary length
hist = [0,0,0,0,0,0,0,0,0,0,0,0]

#for each ID
for i in files:

	#load bytes file
	fp = open("train/"+i[0],"r")
	bytes = fp.read()
	fp.close()

	#clean data
	bytes = re.sub(r'([0-9]|[A-F]){8}',"",bytes)
	bytes = re.sub(r'\n'," ",bytes)
	bytes = re.sub(r'  '," ",bytes)

	#split into list of byte "pixels"
	pixels = [token for token in bytes.split(" ") if token != ""]

	#replace anomalous bytes
	for t in range(len(pixels)):
		if pixels[t] == "??":
			pixels[t] = "00"
		pixels[t] = int(pixels[t], 16)

	l = len(pixels)

	#keep track of max length
	max = 0
	if l > max:
		max = len(pixels)

	#count binary length
	if l >  1024000:
		hist[0]+=1
	elif l > 1000000:
		hist[1]+=1
	elif l > 900000:
		hist[2]+=1
	elif l > 800000:
		hist[3]+=1
	elif l > 700000:
		hist[4]+=1
	elif l > 600000:
		hist[5]+=1
	elif l > 500000:
		hist[6]+=1
	elif l > 400000:
		hist[7]+=1
	elif l > 300000:
		hist[8]+=1
	elif l > 200000:
		hist[9]+=1
	elif l > 100000:
		hist[10]+=1
	else:
		hist[11]+=1

#print stats to batch file in folder "batch_stats"
fp = open("batch_stats/"+sys.argv[1], "w")
fp.write(">1024000: " + str(hist[0])+"\n")
fp.write(">1000000: " + str(hist[1])+"\n")
fp.write(">900000: " + str(hist[2])+"\n")
fp.write(">800000: " + str(hist[3])+"\n")
fp.write(">700000: " + str(hist[4])+"\n")
fp.write(">600000: " + str(hist[5])+"\n")
fp.write(">500000: " + str(hist[6])+"\n")
fp.write(">400000: " + str(hist[7])+"\n")
fp.write(">300000: " + str(hist[8])+"\n")
fp.write(">200000: " + str(hist[9])+"\n")
fp.write(">100000: " + str(hist[10])+"\n")
fp.write("<=100000: " + str(hist[11])+"\n")
fp.write("max: "+ str(max))

print("\n\n\n\t\t\t-> DONE! <-\n\n")





