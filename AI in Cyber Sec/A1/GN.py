
#_________________IMPORTS AND FUNCTION DEFINITIONS

import pandas as pd
import os
import numpy as np
from random import sample
from collections import defaultdict
from scipy.spatial.distance import euclidean, cityblock, mahalanobis

def score(sampVec, *model):
	# sampVec: the sample vector containg keystroke timing information to be classified
	# model: tupel containing 2 components needed to make a classification
	#	model[0] = avgVec: a vector containing the element wise average of the sample vectors obtained from dataset
	#	model[1] = maxD: a vector containing the max distances(mahalanobis, cityblock, euclidean) between avgVec and the other sample vectors for the corresponging subject
	#	
	# returns: a float in (0, 1] where 1 indicates an exact match (i.e. sampVec = avgVec) and values approaching 0 indicate increasing distance between sampVec and avgVec 
	
	#retrien model components
	avgVec = model[0][0]
	maxD = model[0][1]	
	
	#find x where score(x) = threshold
	intersect = 0.176471 - -0.023
	
	#calculate normalized distance measure
	norm = maxD[1]/intersect
	dist = cityblock(sampVec, avgVec) 
	norm_dist = dist/norm

	#score distance
	score = 1/(1+(norm_dist)) 
	return score

def popFitness(pop, *model): #returns a list of scores for each member of pop against *model
	scores = np.zeros((pop.shape[0]))
	for i in range(pop.shape[0]):
		scores[i] = score(pop[i], (model[0][0], model[0][1]))
	return scores

def crossOver(pop, pop_scores, keepPercentage, mutationPercentage):
	
	#calculate number of children to create
	k = int(pop_scores.shape[0] * keepPercentage)
	m = int(pop_scores.shape[0] * mutationPercentage)
	r = pop_scores.shape[0] - k - m
	
	#get indicies of max scores
	max = np.argsort(pop_scores)
	max = np.flip(max)
	
	
	spawn = [] #list of offspring
	for i in range(len(max)-1):
		c1, c2 = mate(pop[max[i]], pop[max[i+1]], pop_scores[max[i]], pop_scores[max[i+1]])
		spawn.append(c1)
		if len(spawn) == r:
			return spawn
		spawn.append(c2)
		if len(spawn) == r:
			return spawn

	print("Not enough offspring created!")
	return spawn

def mate(parent1, parent2, p1score, p2score):
	
	#determine number of genes to swap
	o1 = parent1
	o2 = parent2
	avgScore = (p1score + p2score)/2	
	if avgScore > 0.80:
		swap = 1
	elif avgScore > 0.70:
		swap = 2
	elif avgScore > 0.60:
		swap = 3
	elif avgScore > 0.50:
		swap = 4
	else:
		swap = 5
	
	#generate list of random indicies of length swap	
	ind = sample(range(0,11), swap)
	o1[ind] = parent2[ind]
	o2[ind] = parent1[ind]
	
	return o1, o2

def mutate(pop, pop_scores, mutationPercentage):
	
	
	m = int(pop_scores.shape[0] * mutationPercentage)
	max = np.argsort(pop_scores)[-m:]
	mutes = []
	for i in max:
		
		#determine number of mutations and their strength
		mutant = pop[i]
		s = pop_scores[i]
		if s > 0.80:
			mute = 1
			mod = 0.001
		elif s > 0.70:
			mute = 3
			mod = 0.05
		elif s > 0.60:
			mute = 3
			mod = 0.05
		elif s > 0.50:
			mute = 4
			mod = 0.1
		else:
			mute = 10
			mod = 0.15
		
		#mutatue sample at indicies listed in ind
		ind = sample(range(0,11), mute)
		for i in ind:
			PoM = sample(range(0,2), 1)
			if PoM == [1]:
				mutant[i] - mod
			else:
				mutant[i] + mod
		mutes.append(mutant)
	return mutes
		


def selection(pop, keepPercentage, offspring, mutants):
	
	#get highest scoring chromosomes
	k = int(pop_scores.shape[0] * keepPercentage)
	max = np.argsort(pop_scores)[-k:]
	newpop = pop[max] 
	
	newpop = np.concatenate((newpop, offspring, mutants))
	return newpop
	
#_________________LOAD DATA________________________________________________
avgDF= pd.read_csv("HavgVecs.csv", header=None).to_numpy()
avgVecs = dict()
for i in avgDF:
	avgVecs[i[0]] = i[1:]

maxDF = pd.read_csv("HmaxD.csv", header=None).to_numpy()
maxDs = dict()
for i in maxDF:
	maxDs[i[0]] = i[1:]

#_______________MAIN___________________________________________________

#	CONTROL VARIABLES
pop_size = 10000
target_samp = 's005'
generations = 100
keep = 0.1	#percentage of population to carry over to next generation
mutation = 0.1 #percentage of mutants in each new generation
term_mode = "best_match"
e = 0.01

#	INITIALIZE POPULATION
pop = np.around(np.random.rand(pop_size,11), decimals = 4)
pop_scores = np.zeros((pop_size))

#pop[1]=avgVecs[target_samp]

#	EVOLUTION LOOP
k=0
totalMax = 0
while(k < generations):
	pop_scores = popFitness(pop, (avgVecs[target_samp], maxDs[target_samp]))	#get fitness scores
	offspring = crossOver(pop, pop_scores, keep, mutation)				#get offspring
	mutants = mutate(pop, pop_scores, mutation)					#get mutants
	
	genMax = max(pop_scores)							#get max score 
	genBest = pop[np.where(pop_scores == genMax)[0][0]]				#get max chromosome

	pop = selection(pop, keep, offspring, mutants)					#create next generation
	
	if genMax > totalMax:								#save best chromosome and score
		totalMax = genMax
		totalBest = genBest
		bestGen = k
	print("\nGENERATION: ",k,"\tMAX SCORE: ",totalMax,"\tGEN MAX SCORE:", genMax)	#print stats
	
	#stop if termination conditions met
	if genMax >= 1-e and term_mode == "best_match":
		k = generations
	if genMax >= 0.85 and term_mode == "match":
		k = generations
	
	#optional increase mutation rate near convergence
	#if genMax > 0.9:
	#	mutation = 0.4
	
	k+=1

print("BEST SCORE: ",totalMax,"\nBEST CHROMOSOME:\n",totalBest, "\nBEST GENERATION: ", bestGen)
		