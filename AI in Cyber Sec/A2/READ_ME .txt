x.joblib: The saved models where x is the name of the model

Split.py and Split2.py Used for splitting the dataset into X_train and X_test and X2_test and X2_train

TrainMono.py: Trains the single models and saves them in [model_name].joblib

TrainEnsemble.py: Creates features from output of single models and trains different classification algorithms on them to create an ensemble model

Validate.py: prints balanced accuracy scores of models on validation sets

UnlabeledPredict.py: predicts class for unlabelled samples