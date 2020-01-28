import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
from joblib import dump, load
from keras import layers
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import class_weight

#load and normalize Samples
X = np.load("trainProc/X_U2.npy", allow_pickle=True)/255.

#correct labels to be contigious from 0-6
Y = np.load("trainProc/Y_U2.npy", allow_pickle=True)
for i in range(Y.shape[0]):
	if Y[i] == 1 or Y[i] == 2:
		Y[i] -= 1
	if Y[i] == 4:
		Y[i] -= 2
	if Y[i] == 6:
		Y[i] -= 3
	if Y[i] == 8 or Y[i] == 9:
		Y[i] -= 4

#calculate class weights for trainer
class_weights = class_weight.compute_class_weight('balanced', np.unique(Y), Y)
class_weights = dict(enumerate(class_weights))

#convert to one hot encoding
Y = keras.utils.to_categorical(Y, dtype='float32')

#split out the test se
sss = StratifiedShuffleSplit(n_splits=2, test_size=500)
tt_val_index, _ = sss.split(X, Y)
X_train_val = X[tt_val_index[0]]
Y_train_val = Y[tt_val_index[0]] 
X_test = X[tt_val_index[1]]
Y_test = Y[tt_val_index[1]]

#split out the validation set
sss = StratifiedShuffleSplit(n_splits=2, test_size=100)
tt_index, _ = sss.split(X_train_val, Y_train_val)
X_train = X_train_val[tt_index[0]]
Y_train = Y_train_val[tt_index[0]] 
X_val = X_train_val[tt_index[1]]
Y_val = Y_train_val[tt_index[1]]

#define model
input = layers.Input(shape=(183, 183, 6))
x = layers.Conv2D(128,5,strides=2, kernel_regularizer=regularizers.l2(0.01))(input) #outputs 90x90x128
x = layers.ReLU()(x)
x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
x = layers.Conv2D(256,5,strides=2, kernel_regularizer=regularizers.l2(0.01))(x) #outputs 14x14x128
x = layers.ReLU()(x)
x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
x = layers.Flatten()(x)  #outputs 512
x = layers.Dropout(0.4)(x)
x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
x = layers.Dense(6, activation='softmax')(x)

clf = keras.models.Model(input, x)

clf_optimzer = keras.optimizers.Adam(lr=0.001)

clf.compile(optimizer = clf_optimzer,
			loss='categorical_crossentropy',
			weighted_metrics=['categorical_accuracy'])


#train model________________________________________________________________________________________________________________________
print("\n\nTRAINING:\n")

#save model with minimum validation loss
mcp_save = ModelCheckpoint('NNmodel512.hdf5', save_best_only=True, monitor='val_loss', mode='min')

history = clf.fit(X_train, Y_train,
          batch_size=256,
          epochs=80,
		  class_weight=class_weights,
          verbose=1,
          validation_data=(X_val, Y_val),
		  callbacks=[mcp_save])

#save model train history
dump(history, "history512.joblib")

#exit()

#load train history
history = load("history512.joblib")

#load and compile model
clf = load_model("NNmodel512.hdf5")

clf.compile(optimizer = clf_optimzer,
			loss='categorical_crossentropy',
			weighted_metrics=['categorical_accuracy'])

#calculate and print results________________________________________________________________________________________________________
print("\n\nRESULTS:\n")

#calculate categorical accuracy (unbalanced)
score = clf.evaluate(X_test, Y_test, verbose=0)
Y_pred = clf.predict(X_test, verbose=0)
Y_pred = np.argmax(Y_pred, axis=1)

#calculate confusion matrix
conf_mat = confusion_matrix(np.argmax(Y_test, axis=1), Y_pred)

#print dataset size and split
print("X_train shape:\t"+str(X_train.shape)+"\tY_train shape:\t"+str(Y_train.shape))
print("X_val shape:\t"+str(X_val.shape)+"\tY_val shape:\t"+str(Y_val.shape))
print("X_test shape:\t"+str(X_test.shape)+"\tY_test shape:\t"+str(Y_test.shape))

#print model summary
clf.summary()


#print model validation results
print("Test "+clf.metrics_names[0]+": "+str(score[0]))
print("Test "+clf.metrics_names[1]+": "+str(score[1]))
print("Confusion Matrix:")
print(conf_mat)

#Plot train and validation and loss and categorical accuracy (balanced)_____________________________________________________________
losses = history.history['loss']
val_losses = history.history['val_loss']
cat_accs = history.history['categorical_accuracy']
val_cat_accs = history.history['val_categorical_accuracy']

epochs = range(1, len(losses) + 1)
plt.plot(epochs, losses, 'b', label='Trianing Loss', linestyle='dashed', linewidth=1)
plt.plot(epochs, val_losses, 'b', label='Validation Loss', linewidth=1)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs, cat_accs, 'r', label='Trianing Categorical Accuracy', linestyle='dashed', linewidth=1)
plt.plot(epochs, val_cat_accs, 'r', label='Validation Categorical Accuracy', linewidth=0.5)
plt.xlabel('Epochs')
plt.ylabel('Categorical Accuracy')
plt.legend()
plt.show()

