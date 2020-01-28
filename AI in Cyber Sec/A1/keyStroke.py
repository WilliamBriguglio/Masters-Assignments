
#_________________IMPORTS AND FUNCTION DEFINITIONS

import pandas as pd
import os
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import euclidean, cityblock, mahalanobis

def isMatch(sampVec, *model, mode='e', tol=0.000000):
	# sampVec: the sample vector containg keystroke timing information to be classified
	# model: tupel containing 2-3 components needed to make a classification
	#	model[0] = avgVec: a vector containing the element wise average of the sample vectors obtained from dataset
	#	model[1] = maxD: a vector containing the max distances(mahalanobis, cityblock, euclidean) between avgVec and the other sample vectors for the corresponging subject
	#	model[2] = covar: an optional covariance matrix required if mode='m'
	# mode: a char specifying the distance metric to use( 'e'=euclidean, 'c'=cityblock, 'm'=mahalanobis
	# tol: 0 will make any sample vector with a distance from the acgVec greater then maxD receive a failing score
	#	positive values increases the distance required to receive a failing score (more tolerant)
	#	negative values decrease the distance required to receive a failing score (more strict)
	#	
	# returns: a float in (0, 1] where 1 indicates an exact match (i.e. sampVec = avgVec) and values approaching 0 indicate increasing distance between sampVec and avgVec 
	
	#retrien model components
	avgVec = model[0][0]
	maxD = model[0][1]	
	
	#find x where score(x) = threshold
	intersect = 0.176471 - tol
	
	#calculate normalized distance measure
	if mode == 'e':
		norm = maxD[2]/intersect
		dist = euclidean(sampVec, avgVec) 
		norm_dist = dist/norm
	if mode == 'c':
		norm = maxD[1]/intersect
		dist = cityblock(sampVec, avgVec) 
		norm_dist = dist/norm
	if mode == 'm':
		covar = model[0][2]
		norm = maxD[0]/intersect
		dist = mahalanobis(sampVec, avgVec, covar.T) 
		norm_dist = dist/norm
	
	#score distance
	score = 1/(1+(norm_dist)) 
	return score


#_________________LOAD DATA________________________________________________
filepref = "H" #this controls which dataset to load 
adj = len(filepref)

avgDF= pd.read_csv(filepref+"avgVecs.csv", header=None).to_numpy()
avgVecs = dict()
for i in avgDF:
	avgVecs[i[0]] = i[1:]

maxDF = pd.read_csv(filepref+"maxD.csv", header=None).to_numpy()
maxDs = dict()
for i in maxDF:
	maxDs[i[0]] = i[1:]


files = []
for r, d, f in os.walk(filepref+"covars"):
	for file in f:
		files.append(os.path.join(r, file))

covars = dict()
for i in files:
	if i.__contains__(".csv"):
		covar = pd.read_csv(i, header=None).to_numpy()
		covars[i[7+adj:11+adj]] = covar


files = []
for r, d, f in os.walk(filepref+"samps"):
	for file in f:
		files.append(os.path.join(r, file))

samps = dict()
for i in files:
	if i.__contains__(".csv"):
	    samp = pd.read_csv(i, header=None).to_numpy()
	    samps[i[6+adj:10+adj]] = samp

#_______________MAIN___________________________________________________


n=0 #counts false negatives
t=0 #counts false positives
for key in samps.keys(): #for each subject's set of samples
	arr = samps[key]
	for samp in arr: #for each sample in set
		for key2 in avgVecs.keys(): #for each model
			score = isMatch(samp, (avgVecs[key2], maxDs[key2], covars[key2]), mode='c', tol=0.00039)
			if score < 0.85 and key2 == key:
				t+=1
			if score > 0.85 and key2 != key:
				n+=1
				
wa = ((((520-t)/27040)*(0.5/(520/27040))) + (((26520-n)/27040)*(0.5/(26520/27040))))

print("False negatives:",t)
print("False positives:",n)
print("Weighted accuracy:",wa)


n=0
t=0
tt=0
nt=0
fp = open("FPFN.csv","w") #csv containing FP and FN count for each model
for key2 in avgVecs.keys(): #for each model
	for key in samps.keys(): #for each set of samples
		arr = samps[key]
		for samp in arr: #for each sample in the set
			score = isMatch(samp, (avgVecs[key2], maxDs[key2], covars[key2]), mode='c', tol=-.023)
			if score < 0.85 and key2 == key:
				t+=1
			if score > 0.85 and key2 != key:
				n+=1
	fp.write(key2+","+str(t)+","+str(n)+"\n") #save FP and FN count for a single model
	tt+=t
	nt+=n
	n = 0
	t = 0
fp.close()
print("Total FN:"+str(tt)+"\tTotal FP:"+str(nt))	


n=0
t=0
tt=0
nt=0
i=0
fp = open("FPWolves.csv","w") #csv containing the nukber of successful imitations by each subject
for key in samps.keys(): #for each set of samples
		arr = samps[key]
		for key2 in avgVecs.keys(): #for each sample in the set
			for samp in arr: #for each  model
				score = isMatch(samp, (avgVecs[key2], maxDs[key2], covars[key2]), mode='c', tol=-.023)
				if score > 0.85 and key2 != key: #if false positive
					i+=1
			if i >= 4: #if successful immitation
				n+=1
			i=0
		fp.write(key+","+str(n)+"\n") #save number of successful imitations for a single model
		tt+=t
		nt+=n
		n = 0
		t = 0
fp.close()
print("Total FN:"+str(tt)+"\tTotal FP:"+str(nt))	