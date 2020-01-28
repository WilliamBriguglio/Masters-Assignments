Some folders referred to in this document are not present as they are too large


Prelim:

	train: a folder containing the .bytes and .asm file of each sample, not included here since it is too large

	TrainLabels.csv: These are the labels for each of the malware samples

Collecting Stats: (in the following section, batch refers to batches of raw files from the dataset which are processes in parallel, it does not refer to training batches)

	mkbatch.py: this program reads the list of sample IDs from trainLabels.csv and creates 30 batches and saves 30 batch files in a folder called "batches" each of which contains the IDs of the files which are to be processed in that batch

	getBatchStatsScipt: this is a script which runs parallel instances of getBatchStats.py on the batch files in the folder "batches"
	getBatchStats.py: this program outputs batch files to the folder "batch_stats" each of which containing statistics on the number of files of each size belonging to that batch
	totalStats.py: this program totals the batch stats of the batch files in the folder "batch_stats" and saves it to the file totalStats.csv
	totalStats.csv: contains statistics on the number of files of each size in the dataset, note that the size ranges are slightly different than the size ranges used in the final paper so the stats are slightly different as well.


	getU200934Script: this is a script which runs parallel instances of getU200934.py on the batch files in the folder "batches"
	getU200934.py: this program outputs batch files into the folder "U200934", each batch file contains the IDs of samples with length between 101400 and 200934 bytes



Creating Dataset: (in the following section, batch refers to batches of raw files from the dataset which are processes in parallel, it does not refer to training batches)
	
	createSampleScripts: this is a script which runs parallel instances of createSamples.py on the batch files contained in the folder "U200934"
	createSample.py: this program creates image tensors from the samples in a specific batch in the folder "200934" and saves batch tensors to the folder "SampsU200934"

	combineU2.py: this program combines batch tensors in the folder SampsU200934, and batch files in U200934Samps and outputs the final image tensors for the entire processed dataset as well as a list indicating the IDs of each sample

	createY.py: this program uses the final input tensors and the list indicating the IDs of each sample, both outputted by combineU2.py, in order to create the label vector used by the classifier during training and evaluation



Training and Evaluation:

	ClassifierU2.py: this program trains, saves, and evaluates the model used for classification


	Balanced_Accuracies.png: an image showing the output of balanced_accuracy_score.py
	balanced_accuracy_score.py: this program has the confusion matrices in tables 2,3, and 4 from the report hard coded into it and it outputs their respective balanced accuracy scores computed using a version of the sklearn python library's balanced_accuracy_score() function which was altered to accept confusion matrices as arguments. 


	History512.joblib: This is the keras training history object which can be loaded using the python joblib library

	NNmodel512.hdf5: This is the keras model used for classification, see Visualize.py for code that loads this model

	Arch.png: An summary of the model architecture

	Cat_AccHistory.png and LossHistory.png: a graph of the categorical accuracy and loss during training against the number of epochs

	getImages: gets images of 2 samples from each class

	termoutput: a folder containing 4 images which together show the terminal output when running ClassifierU2.py



Interpretation:

	
	Visualize.py: this program preforms LRP on a select portion of the dataset, contained in X_F5.npy, and prints their relevance maps as well as saves the relevances of their inputs for use by analysis.py

	analysis.py: This is a python script which displays the most relevant 6-grams and their corresponding code. It takes two command line arguments
		sys.argv[1] : an int indicating where to start the ngram analysis, e.g. enter 7 starts at the 7th most important 6-gram  
		sys.argv[1] : an int indicating how many ngrams to analyze, e.g. 5 will print the sys.argv[1]^th most important 6-gram's corresponding code and the next 4 most important 6-gram's corresponding code

	Interpret: a folder containing the following;
		A_2.npy: analysis tensor for a sample 0AnoOZDNbPXIr2MRBSCJ containing the relevance of each of its features
		X_2.npy: imputes tensor for sample 0AnoOZDNbPXIr2MRBSCJ
		X_F5.npy: 5 samples I considered using for the visualization and interpretation section
		0AnoOZDNbPXIr2MRBSCJ.asm: assembly code of the interpreted sample
		0AnoOZDNbPXIr2MRBSCJ.byes: bytes file of the interpreted sample
		0AnoOZDNbPXIr2MRBSCJ_b.png: channels 3-5 of the interpreted sample as an RGB image
		0AnoOZDNbPXIr2MRBSCJ_f.png: channels 0-2 of the interpreted sample as an RGB image
		Terminal output when obtaining code snippets for the 100th and 122nd Mose relevant 6gram
		Rel_map.png: relevance map of the interpreted sample


