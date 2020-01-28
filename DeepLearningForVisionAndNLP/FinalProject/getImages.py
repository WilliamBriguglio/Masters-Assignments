import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

plt.figure()

def display_img(X):	#plots the first 3 channels of X as one RGB immage and the next 3 channels as another RGB image
	print(X.shape)
	plt.axis('off')
	plt.imshow(X[:,:,:3])
	plt.show()
	plt.imshow(X[:,:,3:])
	plt.show()

#load samples and their labels
X = np.load("trainProc/X_U2.npy", allow_pickle=True)/255
Y = np.load("trainProc/Y_U2.npy", allow_pickle=True)

knt = [0]*6
img_per_class = 2

#plot 2 samples of each image
for i in range(Y.shape[0]):
	if Y[i] == 1 and knt[0] < img_per_class:
		print("displaing class:", 1)
		display_img(X[i])
		knt[0]+=1	
	if Y[i] == 2 and knt[1] < img_per_class:
		print("displaing class:", 2)
		display_img(X[i])
		knt[1]+=1	
	if Y[i] == 4 and knt[2] < img_per_class:
		print("displaing class:", 4)
		display_img(X[i])
		knt[2]+=1	
	if Y[i] == 6 and knt[3] < img_per_class:
		print("displaing class:", 6)
		display_img(X[i])
		knt[3]+=1	
	if Y[i] == 8 and knt[4] < img_per_class:
		print("displaing class:", 8)
		display_img(X[i])
		knt[4]+=1	
	if Y[i] == 9 and knt[5] < img_per_class:
		print("displaing class:", 9)
		display_img(X[i])
		knt[5]+=1	

print(knt)
