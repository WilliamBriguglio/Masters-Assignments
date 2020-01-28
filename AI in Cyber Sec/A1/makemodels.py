import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import euclidean, cityblock, mahalanobis

df = pd.read_csv("PWData.csv")

Y = df.to_numpy(dtype=str)[:,0]
X = df.to_numpy()[:,3:]
filepref = ""


DD = [1,4,7,10,13,16,19,22,25,28]
UD = [2,5,8,11,14,17,20,23,26,29]
H = [0,3,6,9,12,15,18,21,24,27,30]
DDUD = [1,2,4,5,7,8,10,11,13,14,16,17,19,20,22,23,25,26,28,29]
DDH = [0,1,3,4,6,7,9,10,12,13,15,16,18,19,21,22,24,25,27,28,30]
UDH =[0,2,3,5,6,8,9,11,12,14,15,17,18,20,21,23,24,26,27,29,30]

USE = H 	#this controls which feature set to use (see arrays on lines 13-18)
filepref = "H"  #this is added to file names so files aren't overwritten when using different feature sets 
X = X[:,USE]	#comment this out to use all features

#the following three dicts use the subject name as key
avgVecs = dict() #dict containing the average vectors
covars = dict() #dict containing covariance matrix
sampArrs = dict() #dict containg list of samples for each subject

cur = Y[0] #get label of first subject "s002"
k = 0
samp=0
for i in range(Y.shape[0]): #for each sample
	if Y[i] == cur: #if sample belongs to current subject
		k += 1
	else:
		#obtain 1st 10 samples from subject [cur] and save to [filepref]samps/[cur]samp.csv
		indices = [i - k + j for j in range(10)] 
		sampArr = np.zeros((10,X.shape[1]))
		for j in range(10):
			sampArr[j] = X[indices[j]]
		fp = open(filepref+"samps/"+cur+"samp.csv","w")
		for row in sampArr:
			for j in range(len(row)-1):
				fp.write("%0.4f," % row[j])
			fp.write("%0.4f\n" % row[j+1])
		fp.close()
		
		#calculate covariance array and average vector for 1st 10 samples from subject [cur]
		sampArrs[cur] = sampArr
		avgVecs[cur] = np.mean(sampArr, axis=0)
		covars[cur] = np.cov(sampArr, rowvar=False)
		cur = Y[i]
		k = 1
		samp+=1

#obatin and save 1st 10 samples from last subject
indices = [i - k + 1 + j for j in range(10)]
sampArr = np.zeros((10,X.shape[1]))
for j in range(10):
	sampArr[j] = X[indices[j]]
fp = open(filepref+"samps/"+cur+"samp.csv","w")
for row in sampArr:
	for j in range(len(row)-1):
		fp.write("%.4f," % row[j])
	fp.write("%.4f\n" % row[j+1])
fp.close()
sampArrs[cur] = sampArr
avgVecs[cur] = np.mean(sampArr, axis=0)
covars[cur] = np.cov(sampArr, rowvar=False)


#save list of average vectors
fp = open(filepref+"avgVecs.csv","w")
for key in avgVecs.keys():
	vec = avgVecs[key]
	fp.write(key+",")
	for j in range(len(vec)-1):
		fp.write(str(round(vec[j],4))+",")
	fp.write(str(round(vec[j+1],4))+"\n")
fp.close()

#save covariance arrays
k = 0
for key in covars.keys():
	arr = covars[key]
	fp = open(filepref+"covars/"+key+"covars.csv","w")
	for row in arr:
		for i in range(len(row)-1):
			fp.write("%f," % row[i])
		fp.write("%f\n" % row[i+1])
	fp.write("\n")
	fp.close()
	k+=1

#for each subject, calculate and save a list of max distances from avgvec to each sample
fp = open(filepref+"maxD.csv","w")
for key in avgVecs.keys():
	m = 0
	c = 0
	e = 0
	for samp in sampArrs[key]:
		m = max(m, mahalanobis(samp, avgVecs[key], covars[key].T))
		c = max(c, cityblock(samp, avgVecs[key]))
		e = max(e, euclidean(samp, avgVecs[key]))
	fp.write(key+","+str(m)+","+str(c)+","+str(e)+"\n")
fp.close
