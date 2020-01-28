import numpy as np

def balanced_accuracy_score(C, sample_weight=None, adjusted=False):
	#Code taken from scikit learn's balanced_accuracy_score in order to make it accept confusion matricies

	with np.errstate(divide='ignore', invalid='ignore'):
		per_class = np.diag(C) / C.sum(axis=1)
	if np.any(np.isnan(per_class)):
		warnings.warn('y_pred contains classes not in y_true')
		per_class = per_class[~np.isnan(per_class)]
	score = np.mean(per_class)
	if adjusted:
		n_classes = len(per_class)
		chance = 1 / n_classes
		score -= chance
		score /= 1 - chance
	return score

#define confusion matricies for the three models based of tables 2,3, and 4 from the report
OurModel = np.array([[149,0,0,1,0,0],
			[0,16,0,0,0,0],
			[1,0,44,0,0,0],
			[0,0,0,32,1,1],
			[2,0,0,0,190,0],
			[1,0,0,0,0,62]])

Chen =     np.array([[154,0,0,0,3,0],
			[0,238,0,0,3,1],
			[1,0,33,1,0,0],
			[1,0,0,63,1,0],
			[2,0,0,0,119,0],
			[0,4,0,0,0,102]])

Gilbert =  np.array([[1490,4,2,9,28,3],
			[6,2440,0,7,8,16],
			[3,0,461,1,3,2],
			[8,6,2,713,10,9],
			[44,4,8,17,1138,8],
			[2,2,0,6,5,996]])

#define test confusion matrix where balanced accuracy is known
test = np.array([[3, 1],
 		[1, 1]])

print("Balanced Accuracy Scores:")
print("Test:",balanced_accuracy_score(test),"--->Expected 0.625")
print("OurModel:",balanced_accuracy_score(OurModel))
print("Chen:",balanced_accuracy_score(Chen))
print("Gilbert:",balanced_accuracy_score(Gilbert))



