
#William Briguglio
#The following code is completely my own.

#_________________________IMPORTS________________________________________________________
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import datetime
import numpy as np
import math
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preproc
#_________________________FUNCTIONS______________________________________________________

def loadData():
	fileName = 'URLdata.csv'
	dataframe = pd.read_csv(fileName, delimiter = ',')
	Xnames= list(dataframe.columns.values)[1:]
	data = (np.array(dataframe))
	Y = data[:,20]
	X = data[:,1:20]
	
	print("Data loaded!")		
	
	return X, Y, Xnames

def daysDiff(first, second): #returns an int value equal the number of days between second and first 
    first = str.split(first)[0] #remove time of day
    
    if(len(first) == 9):#pad single digit day of month with a 0
	    first = "0" + first 

    format = "%d/%m/%Y"
    firstDateObj = datetime.datetime.strptime(first, format)
    secondDateObj = datetime.datetime.strptime(second, format)
    timeDelta = secondDateObj - firstDateObj
    return timeDelta.days

#_________________________MAIN____________________________________________________________	

CURR_DATE = '18/02/2019'

X, Y, Xnames = loadData()#Obtain Labels(Y) and Features(X) and column names Xnames
#X containes 1781 rows of 19 features
#Y containes 1781 labels where 1 = malicious and 0 = benign

#delete rows 1306 and 1659 which have a value of NaN
X = np.delete(X, (1659, 1306), axis=0)
Y = np.delete(Y, (1659, 1306), axis=0)

#separate numerical and categorical features
Xcat = X[:,[2,3,5,6]]#four categorical features
Xnum = X[:,[0,1,4,9,10,11,12,13,14,15,16,17,18]]#thirteen numerical features

XnumNames = [Xnames[i] for i in [0,1,4,9,10,11,12,13,14,15,16,17,18]]# column names for numerical features

#-----Converting dates to numerical features---------------------------------------------
dateFeatures = np.zeros((len(Xnum[:,8]), 2))
dateNames = np.array(["daysSinceReg", "daysSinceUpdate"])
#col 0 = days since registration
#col 1 - days since last update

for index in range(len(X[:,7])):
	if len(X[index, 7]) > 6:
	    dateFeatures[index, 0] = daysDiff(X[index,7], CURR_DATE) #convert registration date to days since registration
	if len(X[index, 8]) > 6:
	    dateFeatures[index, 1] = daysDiff(X[index,8], CURR_DATE) #convert last update date to days since last update
	#invalid values remain 0

Xnum = np.concatenate((Xnum, dateFeatures), axis=1) #add date features to numerical features
XnumNames = np.concatenate((XnumNames, dateNames)) #update numerical feature column names

#-----pre-processing numerical features--------------------------------------------------
for index in range(len(Xnum[:,2])):#if content_length equals NaN, set content_length to 0 
    if math.isnan(Xnum[index,2]):
	    Xnum[index,2] = 0
	    
Y = Y.astype('int')
Xnum = Xnum.astype('int')
Xnum = preproc.normalize(Xnum, norm='l2')
Xnum = preproc.scale(Xnum)

#-----pre-processing categorical features------------------------------------------------

#consistently label iso-8859-1 and utf-8 with uppercase chars
for index in range (len(Xcat[:,0])):
	if Xcat[index,0] == "iso-8859-1" or Xcat[index,0] == "utf-8":
		Xcat[index,0] = (Xcat[index,0].upper())

#create feature that corresponds to type of server without version number
serverSuperFeatures = np.chararray(len(Xcat[:,1]), unicode=True, itemsize=10)
for index in range (len(Xcat[:,1])):
	tkz = str.split(Xcat[index,1], '/')[0]
	tkz = str.split(tkz, '-')[0]
	tkz = str.split(tkz, '2')[0]
	if tkz == 'nginx' or tkz == 'Apache' or tkz == 'Microsoft' or tkz == 'mw':
	    serverSuperFeatures[index] = tkz 
	else:
	    serverSuperFeatures[index] = 'misc'
	     
Xcat = np.column_stack((Xcat, serverSuperFeatures))

#consistently label locations
for index in range (len(Xcat[:,2])):
	if Xcat[index,2] == "United Kingdom" or Xcat[index,2] == "[u'GB'; u'UK']":
		Xcat[index,2] = "UK"
	if Xcat[index,2] == "ru" or Xcat[index,2] == "se" or Xcat[index,2] == "us":
		Xcat[index,2] = (Xcat[index,2].upper())
		
XcatAug = Xcat[:,[0,1,2,4]] #remove whois_statepro
enc = preproc.OneHotEncoder()
enc.fit(XcatAug)
XencNames = enc.get_feature_names()
Xenc = enc.transform(XcatAug).toarray()

#-----recombine categorical and numerical features---------------------------------------

X2 = np.concatenate((Xenc, Xnum), axis=1)
X2Names = np.concatenate((XencNames, XnumNames))

#-----Save data--------------------------------------------------------------------

#X2 features with categorical features as one hot encoded
#Y Lbales with 0 = benign, 1 = malicious

print("X Names shape: ",X2Names.shape)
print("X shape: ",X2.shape)

print("Y shape: ",Y.shape)

x_train, x_test, y_train, y_test = train_test_split(X2, Y, test_size=0.33)


with open("x_test.csv", "w+") as fp:
	for x_name in X2Names:
		fp.write(x_name+",")
	fp.write("\n")
	for x in x_test:
		for i in range(len(x)):
			if i != (len(x) - 1):
				fp.write(str(x[i])+",")
			else:
				fp.write(str(x[i]))
		fp.write("\n ")

with open("y_test.csv", "w+") as fp:
	fp.write("Labels\n")
	for y in y_test:
		fp.write(str(y)+"\n")


with open("x_train.csv", "w+") as fp:
	for x_name in X2Names:
		fp.write(x_name+",")
	fp.write("\n")
	for x in x_train:
		for i in range(len(x)):
			if i != (len(x) - 1):
				fp.write(str(x[i])+",")
			else:
				fp.write(str(x[i]))
		fp.write("\n ")

with open("y_train.csv", "w+") as fp:
	fp.write("Labels\n")
	for y in y_train:
		fp.write(str(y)+"\n")


exit()
with open("ProcURLData.csv", "w+") as fp:
	for x_name in X2Names:
		fp.write(x_name+",")
	fp.write("\n")
	for x in X2:
		for i in range(len(x)):
			if i != (len(x) - 1):
				fp.write(str(x[i])+",")
			else:
				fp.write(str(x[i]))
		fp.write("\n ")

with open("ProcURLLabels.csv", "w+") as fp:
	fp.write("Labels\n")
	for y in Y:
		fp.write(str(y)+"\n")