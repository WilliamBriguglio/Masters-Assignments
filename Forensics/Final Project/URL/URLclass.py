
#_________________________IMPORTS________________________________________________________

import multiprocessing
import pandas as pd
import numpy as np
import math
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.feature_selection import RFE, f_classif, mutual_info_classif
from sklearn import linear_model

import keras
import keras.backend
import keras.models
from keras.optimizers import Adam
from keras.initializers import TruncatedNormal as TN
from keras.initializers import Constant
from keras.models import model_from_json

import innvestigate

#_________________________FUNCTIONS______________________________________________________

def loadData():
	fileName = 'ProcURLData.csv'
	df = pd.read_csv(fileName, delimiter = ',')
	X = df.to_numpy()[:,:310]
	
	X_names = np.array(list(df.columns.values)[:310])
	
	fileName = 'ProcURLLabels.csv'
	df = pd.read_csv(fileName, delimiter = ',')
	Y = df.to_numpy()
	
	return X, Y, X_names

def loadTest():
	fileName = 'x_test.csv'
	df = pd.read_csv(fileName, delimiter = ',')
	X = df.to_numpy()[:,:310]
	
	X_names = np.array(list(df.columns.values)[:310])
	
	fileName = 'y_test.csv'
	df = pd.read_csv(fileName, delimiter = ',')
	Y = df.to_numpy()
	
	return X, Y, X_names

def loadTrain():
	fileName = 'x_train.csv'
	df = pd.read_csv(fileName, delimiter = ',')
	X = df.to_numpy()[:,:310]
	
	X_names = np.array(list(df.columns.values)[:310])
	
	fileName = 'y_train.csv'
	df = pd.read_csv(fileName, delimiter = ',')
	Y = df.to_numpy()
	
	return X, Y, X_names

def Save1DArray(A, filename):
	with open(filename+".csv", "w+") as fp:
		for a in A:
		    fp.write(str(a)+"\n")

#_________________________MAIN____________________________________________________________	

#FEATURE SELECTION
x_test, y_test, x_names = loadTest()
x_train, y_train, x_names = loadTrain()
F_scores, Pvals = f_classif(x_train, y_train)
#Save1DArray(F_scores, "F_scores")

MI_scores = mutual_info_classif(x_train, y_train)#mutual information
#Save1DArray(MI_scores, "MI_scores")


forest = RFC(n_estimators=100, class_weight='balanced', max_depth=None, n_jobs=-1)
LR = linear_model.LogisticRegression(solver='liblinear', tol=0.0000001, C = 1000)
rfe = RFE(LR, 1, step=1, verbose=1)
RFE_scores = rfe.fit(x_train, y_train)
RFE_rankings = np.array(RFE_scores.ranking_)
#Save1DArray(RFE_rankings, "RFE_rankings")

#---------------------TRAINING and TESTING-------------------------------

#extract selected features
for i in range(len(F_scores)):
	if math.isnan(F_scores[i]):
		F_scores[i] = -1
maxIs = np.argsort(F_scores)
F_names = x_names[maxIs[-20:]]
x_F_train = x_train[:,maxIs[-20:]]
x_F_test = x_test[:,maxIs[-20:]]

maxIs = np.argsort(MI_scores)
MI_names = x_names[maxIs[-20:]]
x_MI_train = x_train[:,maxIs[-20:]]
x_MI_test = x_test[:,maxIs[-20:]]

maxIs = np.argsort(RFE_rankings)
ax_RFE = RFE_rankings[maxIs[:20]]
RFE_names = x_names[maxIs[:20]]
x_RFE_train = x_train[:,maxIs[:20]]
x_RFE_test = x_test[:,maxIs[:20]]

fileName = '20MostAct.csv'
df = pd.read_csv(fileName, delimiter = ',')
maxIs = df.to_numpy()
LRP_names = x_names[maxIs[-20:]]
x_LRP_train = x_train[:,maxIs[-20:]]
x_LRP_test = x_test[:,maxIs[-20:]]
x_LRP_train = np.squeeze(x_LRP_train)
x_LRP_test = np.squeeze(x_LRP_test)
 
f = open("Results4.txt","w+")

#initialize classifiers
LogReg = linear_model.LogisticRegression(solver='liblinear', tol=0.0000001, C = 1000)
RF = RFC(n_estimators=100, class_weight='balanced', max_depth=None, n_jobs=-1)
NB = GaussianNB()
SVM = SVC(probability=False, kernel = 'poly' , class_weight='balanced', C = 1, gamma=1, degree=3)

"""
#Uncomment this code to create and initialize a new NN and save it
model = keras.models.Sequential([
    keras.layers.Dense(20, kernel_initializer=TN(stddev=0.1),bias_initializer=Constant(0.1)),
    keras.layers.Dense(10, activation="relu", kernel_initializer=TN(stddev=0.1),bias_initializer=Constant(0.1)),
    keras.layers.Dropout(0.45),
    keras.layers.Dense(1, activation="sigmoid", kernel_initializer=TN(stddev=0.1), bias_initializer=Constant(0.1)),
])
model.compile(loss="binary_crossentropy",
		optimizer=Adam(lr=0.001, epsilon=0.0001),
		metrics=["binary_accuracy"])

#Save initialized model since initial values are random
model_json = model.to_json()
with open("NNmodel2.json", "w") as json_file:
    json_file.write(model_json)
    
# serialize weights to HDF5
model.save_weights("NNweights2.h5")
print("Saved model to disk")
"""
#------------------Neural Network

f.write("\nNueral Network\n")

_BS =25 
_E = 15

json_file = open('NNmodel2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("NNweights2.h5")
model.compile(loss="binary_crossentropy",
		optimizer=Adam(lr=0.001, epsilon=0.0001),
		metrics=["binary_accuracy"])

#F_score
f.write("Fscore\n")
history = model.fit(x_F_train, y_train, batch_size=_BS, epochs=_E, verbose=0)
y_pred = model.predict_classes(x_F_test)
f.write(str(confusion_matrix(y_test, y_pred))+"\n")
f.write("bal. acc.:"+str(balanced_accuracy_score(y_test, y_pred))+"\n")


"""#un comment this code for LRP
#Separate samples classified as benign and samples classified as malicious
m = 0 #predicted malicious count
b = 0 #predicted benign count
for y in y_pred:
	if y == 1:
	    m+=1
	else:
	    b+=1

c = 0
d = 0
x_mal = np.zeros((m,20))
y_mal = np.zeros((m))
x_benign = np.zeros((b,20))
y_benign = np.zeros((b))
for i in range(len(y_test)):
	if y_pred[i] == [1]:
		x_mal[c] = x_F_test[i]
		y_mal[c] = y_test[i]
		c+=1
	else:
		x_benign[d] = x_F_test[i]
		y_benign[d] = y_test[i]
		d+=1

Save1DArray(x_mal[0], "x_m_ex")
#test_samples = list(zip(x_benign[:], y_benign[:]))
test_samples = list(zip(x_mal[:], y_mal[:]))
analysis = np.zeros([len(test_samples), 20])

analyzer = innvestigate.create_analyzer("lrp.z", model)

for i, (x, y) in enumerate(test_samples):
    # Add batch axis.
    x = x[None, :]
    
    # Predict final activations, probabilites, and label.
    y_hat = model.predict_classes(x)[0]
    
    a = analyzer.analyze(x)
    a = a.astype(np.float32)
    
    #SaveActivations(a[0], "Act_Max_"+str(y[0])+"_"+str(y_hat[0]))
    # Store the analysis.
    analysis[i]  = a[0]

print(analysis[0])
exit()
mean_abs_act = np.mean(analysis, axis=0)
Save1DArray(mean_abs_act, "Act_Abs_Mean_Benign_20")
exit()
#"""

json_file = open('NNmodel2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("NNweights2.h5")
model.compile(loss="binary_crossentropy",
		optimizer=Adam(lr=0.001, epsilon=0.0001),
		metrics=["binary_accuracy"])

#Mutual Information
f.write("Mutual Information\n")
history = model.fit(x_MI_train, y_train, batch_size=_BS, epochs=_E, verbose=0)
y_pred = model.predict_classes(x_MI_test)
f.write(str(confusion_matrix(y_test, y_pred))+"\n")
f.write("bal. acc.:"+str(balanced_accuracy_score(y_test, y_pred))+"\n")

json_file = open('NNmodel2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("NNweights2.h5")
model.compile(loss="binary_crossentropy",
		optimizer=Adam(lr=0.001, epsilon=0.0001),
		metrics=["binary_accuracy"])

#RFE
f.write("RFE\n")
history = model.fit(x_RFE_train, y_train, batch_size=_BS, epochs=_E, verbose=0)
y_pred = model.predict_classes(x_RFE_test)
f.write(str(confusion_matrix(y_test, y_pred))+"\n")
f.write("bal. acc.:"+str(balanced_accuracy_score(y_test, y_pred))+"\n")

json_file = open('NNmodel2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("NNweights2.h5")
model.compile(loss="binary_crossentropy",
		optimizer=Adam(lr=0.001, epsilon=0.0001),
		metrics=["binary_accuracy"])

#LRP
f.write("LRP\n")
history = model.fit(x_LRP_train, y_train, batch_size=_BS, epochs=_E, verbose=0)
y_pred = model.predict_classes(x_LRP_test)
f.write(str(confusion_matrix(y_test, y_pred))+"\n")
f.write("bal. acc.:"+str(balanced_accuracy_score(y_test, y_pred))+"\n")
exit()
#-------------SVM
f.write("Support Vector Machine\n")

#grid search params C=1000, degree=4, gamma=10
params = {'C':[0.10,1, 10, 100, 1000],'gamma':[0.001,0.01,0.1,1,10], 'degree':[2,3,4]}

#F_score
f.write("Fscore\n")
GS = GridSearchCV(SVM, params, scoring='balanced_accuracy', cv=2, n_jobs=-1, verbose=2)
GS.fit(x_F_train, y_train)
SVM.set_params(**(GS.best_params_))
SVM.fit(x_F_train, y_train)
y_pred = SVM.predict(x_F_test)
f.write(str(GS.best_params_)+"\n")
f.write(str(confusion_matrix(y_test, y_pred))+"\n")
f.write("bal. acc.:"+str(balanced_accuracy_score(y_test, y_pred))+"\n")


#Mutual Information
f.write("Mutual Information\n")
GS = GridSearchCV(SVM, params, scoring='balanced_accuracy', cv=2, n_jobs=-1, verbose=2)
GS.fit(x_MI_train, y_train)
SVM.set_params(**(GS.best_params_))
SVM.fit(x_MI_train, y_train)
y_pred = SVM.predict(x_MI_test)
f.write(str(GS.best_params_)+"\n")
f.write(str(confusion_matrix(y_test, y_pred))+"\n")
f.write("bal. acc.:"+str(balanced_accuracy_score(y_test, y_pred))+"\n")

#RFE

params = {'C':[0.10,1, 10, 100, 1000],'gamma':[0.001,0.01,0.1,1], 'degree':[4]}
f.write("RFE\n")
GS = GridSearchCV(SVM, params, refit=False, scoring='balanced_accuracy', cv=2, n_jobs=1, verbose=3, error_score=0.0)
GS.fit(x_RFE_train, y_train)
SVM.set_params(**(GS.best_params_))
SVM.fit(x_RFE_train, y_train)
y_pred = SVM.predict(x_RFE_test)
f.write(str(GS.best_params_)+"\n")
f.write(str(confusion_matrix(y_test, y_pred))+"\n")
f.write("bal. acc.:"+str(balanced_accuracy_score(y_test, y_pred))+"\n")


#LRP
f.write("LRP\n")
GS = GridSearchCV(SVM, params, scoring='balanced_accuracy', cv=2, n_jobs=-1, verbose=2)
GS.fit(x_LRP_train, y_train)
SVM.set_params(**(GS.best_params_))
SVM.fit(x_LRP_train, y_train)
y_pred = SVM.predict(x_LRP_test)
f.write(str(GS.best_params_)+"\n")
f.write(str(confusion_matrix(y_test, y_pred))+"\n")
f.write("bal. acc.:"+str(balanced_accuracy_score(y_test, y_pred))+"\n")


#------------Naive Bayes
f.write("\nNaive Bayes\n")

#F_score
f.write("Fscore\n")
NB.fit(x_F_train, y_train)
y_pred = NB.predict(x_F_test)
f.write(str(confusion_matrix(y_test, y_pred))+"\n")
f.write("bal. acc.:"+str(balanced_accuracy_score(y_test, y_pred))+"\n")

#Mutual Information
f.write("Mutual Information\n")
NB.fit(x_MI_train, y_train)
y_pred = NB.predict(x_MI_test)
f.write(str(confusion_matrix(y_test, y_pred))+"\n")
f.write("bal. acc.:"+str(balanced_accuracy_score(y_test, y_pred))+"\n")

#RFE
f.write("RFE\n")
NB.fit(x_RFE_train, y_train)
y_pred = NB.predict(x_RFE_test)
f.write(str(confusion_matrix(y_test, y_pred))+"\n")
f.write("bal. acc.:"+str(balanced_accuracy_score(y_test, y_pred))+"\n")

#LRP
f.write("LRP\n")
NB.fit(x_LRP_train, y_train)
y_pred = NB.predict(x_LRP_test)
f.write(str(confusion_matrix(y_test, y_pred))+"\n")
f.write("bal. acc.:"+str(balanced_accuracy_score(y_test, y_pred))+"\n")


#------------Random Forest
f.write("\nRandom Forest\n")

#F_score
f.write("Fscore\n")
RF.fit(x_F_train, y_train)
y_pred = RF.predict(x_F_test)
f.write(str(confusion_matrix(y_test, y_pred))+"\n")
f.write("bal. acc.:"+str(balanced_accuracy_score(y_test, y_pred))+"\n")

#Mutual Information
f.write("Mutual Information\n")
RF.fit(x_MI_train, y_train)
y_pred = RF.predict(x_MI_test)
f.write(str(confusion_matrix(y_test, y_pred))+"\n")
f.write("bal. acc.:"+str(balanced_accuracy_score(y_test, y_pred))+"\n")

#RFE
f.write("RFE\n")
RF.fit(x_RFE_train, y_train)
y_pred = RF.predict(x_RFE_test)
f.write(str(confusion_matrix(y_test, y_pred))+"\n")
f.write("bal. acc.:"+str(balanced_accuracy_score(y_test, y_pred))+"\n")

f.write("LRP\n")
RF.fit(x_LRP_train, y_train)
y_pred = RF.predict(x_LRP_test)
f.write(str(confusion_matrix(y_test, y_pred))+"\n")
f.write("bal. acc.:"+str(balanced_accuracy_score(y_test, y_pred))+"\n")


#------------Logistic Regression
f.write("\nLogistic Regression\n")

#grid search params
params = {'C':[0.10,1, 10, 100, 1000]}

#F_score
f.write("Fscore\n")
GS = GridSearchCV(LogReg, params, scoring='balanced_accuracy', cv=2, n_jobs=-1, verbose=2)
GS.fit(x_F_train, y_train)
LogReg.set_params(**(GS.best_params_))
LogReg.fit(x_F_train, y_train)
y_pred = LogReg.predict(x_F_test)
f.write(str(GS.best_params_)+"\n")
f.write(str(confusion_matrix(y_test, y_pred))+"\n")
f.write("bal. acc.:"+str(balanced_accuracy_score(y_test, y_pred))+"\n")

#Mutual Information
f.write("Mutual Information\n")
GS = GridSearchCV(LogReg, params, scoring='balanced_accuracy', cv=2, n_jobs=-1, verbose=2)
GS.fit(x_MI_train, y_train)
LogReg.set_params(**(GS.best_params_))
LogReg.fit(x_MI_train, y_train)
y_pred = LogReg.predict(x_MI_test)
f.write(str(GS.best_params_)+"\n")
f.write(str(confusion_matrix(y_test, y_pred))+"\n")
f.write("bal. acc.:"+str(balanced_accuracy_score(y_test, y_pred))+"\n")

#RFE
f.write("RFE\n")
GS = GridSearchCV(LogReg, params, refit=False, scoring='balanced_accuracy', cv=2, n_jobs=1, verbose=3, error_score=0.0)
if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver', force=True)
    GS.fit(x_RFE_train, y_train)
LogReg.set_params(**(GS.best_params_))
LogReg.fit(x_RFE_train, y_train)
y_pred = LogReg.predict(x_RFE_test)
f.write(str(GS.best_params_)+"\n")
f.write(str(confusion_matrix(y_test, y_pred))+"\n")
f.write("bal. acc.:"+str(balanced_accuracy_score(y_test, y_pred))+"\n")

#LRP
f.write("LRP\n")
GS = GridSearchCV(LogReg, params, scoring='balanced_accuracy', cv=2, n_jobs=-1, verbose=2)
GS.fit(x_LRP_train, y_train)
LogReg.set_params(**(GS.best_params_))
LogReg.fit(x_LRP_train, y_train)
y_pred = LogReg.predict(x_LRP_test)
f.write(str(GS.best_params_)+"\n")
f.write(str(confusion_matrix(y_test, y_pred))+"\n")
f.write("bal. acc.:"+str(balanced_accuracy_score(y_test, y_pred))+"\n")



