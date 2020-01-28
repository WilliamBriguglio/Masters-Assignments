import innvestigate
import pandas as pd
import numpy as np
import keras
import sys
import matplotlib.pyplot as plt
from joblib import dump, load
from keras import layers
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import class_weight
import innvestigate.utils as iutils

#load tensor containing just 5 samples
X = np.load("Interpret/X_F5.npy", allow_pickle=True)/255

#create label array (I know already that X_F5 has samples all belong int to class 0)
Y = np.array([0, 0, 0, 0, 0, 1, 2, 3, 4, 5]) #add extra labels (1,2,3,4,5) so to_categorical knows to make one hot encoded target vector with 6 classes
Y = keras.utils.to_categorical(Y, dtype='float32')[:5] #remove fake labels (1,2,3,4,5)
Y_lbls = np.ones(X.shape[0]) 


#load and compile model
clf = load_model("NNmodel512.hdf5", compile=False)

clf_optimzer = keras.optimizers.Adam(lr=0.001)
clf.compile(optimizer = clf_optimzer,
			loss='categorical_crossentropy',
			weighted_metrics=['categorical_accuracy'])
#print model summary
clf.summary()

#make predictions and print out their confidence
y_pred = clf.predict(X)
classes = [1,2,4,6,8,9]
for i in y_pred:
	print("Predicted Class", classes[np.argmax(i)], "with probability", np.max(i))

#rename since innvestiaget kept throwing an error about too many layers called 'dense_1'
for i in clf.layers:
	if i.name == 'dense_1':
		i.name = 'dense_0'

#initialize LRP analyzer
clf = innvestigate.utils.model_wo_softmax(clf)
clf.summary()
analyzer = innvestigate.create_analyzer("lrp.epsilon", clf)

#zip all samples with their label
n = len(X)
s = 0
test_samples = list(zip(X[s:n], Y_lbls[s:n]))

#initialize array of zeros to store analysis after it is completed
analysis = np.zeros([len(test_samples), 183, 183, 6])


#run analysis for all samples inn test_samples
for i, (x, y) in enumerate(test_samples):
	#add batch axis
	x = x[None, :]

	#analyze sample
	a = analyzer.analyze(x)
	a = a.astype(np.float32)

	#add sample's analysis to list of analysis results
	analysis[i]  = a[0]
		
	#print prediction results
	print("True class:",y,"prediction",classes[np.argmax(clf.predict(x))])

#save analysis
np.save("Interpret/A_F5.npy", analysis)
analysisAvg = np.average(analysis, axis=3)

#get max activation, used to specify range for the colour map later
maxAct = max(np.max(analysisAvg), -np.min(analysisAvg))

#show visualizatons for sample 3 of 5
i == 2 #sampl with 99.99% prob of being class 1 (its true class) ID: 0AnoOZDNbPXIr2MRBSCJ
plt.figure()

plt.imshow(X[i,:,:,:3])
plt.show()
plt.imshow(X[i,:,:,3:])
plt.show()
plt.imshow(analysisAvg[i], cmap='seismic', vmin=-maxAct, vmax=maxAct)
plt.show()