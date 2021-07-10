import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
from helper import *
import numpy as np
import random

######################### GA methods ################################
# constraint: parent x2(t) cannot be worse than parent x1(t).
def crossover1(p1,p2, mutationRate):
	x1 = 0
	x2 = 0
	if funcOne(p2) >= funcOne(p1):
		x2 = p2
		x1 = p1
	else:
		x2 = p1 
		x1 = p2		
	child1 = random.uniform(0,1) * (x2 - x1) + x2
	child2 = random.uniform(0,1) * (x2 - x1) + x2
	child1 = clipBound(child1)
	child2 = clipBound(child2)
	return child1, child2

def crossover2(parent1,parent2, mutationRate):
	c1x, c1y = crossover1(parent1[0], parent2[0], mutationRate)
	c2x, c2y = crossover1(parent1[1], parent2[1], mutationRate)
	return c1x,c1y,c2x,c2y

def clipBound(x):
	if x > 5:
		overflow = x / 5
		return 5 - (1 - sigmoid(x))
	if x < -5:
		overflow = abs(x) / 5
		return (5 - (1 - sigmoid(x))) * -1
	return x

def fitness(maxVal,minVal, y):
	return (y - minVal) / (maxVal - minVal)

def calcFitness(pop):
	y = np.zeros(len(pop))
	for i in range(len(pop)):
		y[i] = funcOne(pop[i])
	maxVal = np.amax(y)
	minVal = np.amin(y)
	fitness_arr = fitness(maxVal, minVal, y)
	df = np.column_stack((pop,y,fitness_arr))
	# sort by fittness value
	df = df[np.argsort(df[:, 2])]
	return df

def calcFitness2(pop):
	z = np.zeros(len(pop))
	for i in range(len(pop)):
		z[i] = funcTwo(pop[i][0], pop[i][1])
	maxVal = np.amax(z)
	minVal = np.amin(z)
	fitness_arr = fitness(maxVal, minVal, z)
	df = np.column_stack((pop,z,fitness_arr))
	# sort by fittness value
	df = df[np.argsort(df[:, 3])]
	return df

# makes next generation from random combination of parents: function 1
def replicate(parents, mutationRate):
	# shuffle so each combination of parents is random
	random.shuffle(parents)
	children = []
	i = 0
	while i < (len(parents) - 1):
		p1 = parents[i] 
		p2 = parents[i+1]
		c1, c2 = crossover1(p1,p2, mutationRate)
		# Mutation	
		if random.uniform(0,1) <= mutationRate:
			c1 = np.random.uniform(-5,5)
		if(random.uniform(0,1) <= mutationRate):
			c2 = np.random.uniform(-5,5)
		children.append(c1)
		children.append(c2)
		i = i + 2
	return children

# makes next generation from random combination of parents: function 2
def replicate2(parents, mutationRate):
	# shuffle so each combination of parents is random
	random.shuffle(parents)
	i = 0
	while i < (len(parents) - 1):
		parent1 = parents[i,:]
		parent2 = parents[i + 1,:]
		# Cross Over
		c1x,c1y,c2x,c2y = crossover2(parent1, parent2, mutationRate)
		childOne = np.array([c1x,c1y])
		childTwo = np.array([c2x,c2y])
		# Mutation
		if random.uniform(0,1) <= mutationRate:
			childOne[0] = np.random.uniform(-5,5)
		if random.uniform(0,1) <= mutationRate:
			childOne[1] = np.random.uniform(-5,5)
		if random.uniform(0,1) <= mutationRate:
			childTwo[0] = np.random.uniform(-5,5)
		if random.uniform(0,1) <= mutationRate:
			childTwo[1] = np.random.uniform(-5,5)
		parents[i] = childOne
		parents[i + 1] = childTwo
		i = i + 2
	return parents

######################### Hill Climbing ################################
def hillClimbFunc2(step_size):
	num_steps = 0
	startingPointX = np.random.uniform(-5,5)
	startingPointY = np.random.uniform(-5,5)
	x = startingPointX
	y = startingPointY
	previous_x = x
	previous_y = y
	start = True
	while funcTwo(x, y) >  funcTwo(previous_x, previous_y) or start:	
		start = False
		previous_x = x
		previous_y = y
		z = funcTwo(x, y)
		if x <= 5 and funcTwo(x + step_size, y) >= z:
			x = x + step_size
		elif x >= -5 and funcTwo(x - step_size, y) >= z:
			x = x - step_size
		elif y <= 5 and funcTwo(x, y + step_size) >= z:
			y = y + step_size
		elif y >= -5 and funcTwo(x, y - step_size) >= z:
			y = y - step_size
	return funcTwo(previous_x, previous_y), round(previous_x,2), round(previous_y,2)

def hillClimbFunc1(step_size):
	num_steps = 0
	startingPoint = np.random.uniform(-5,5)
	x = startingPoint
	previous = x
	while funcOne(x) >=  funcOne(previous) and x <= 5 and x >= -5:	
		previous = x
		if funcOne(x + step_size) >= funcOne(x - step_size):
			x = x + step_size
		else:
			x = x - step_size
		num_steps = num_steps + 1
	print("steps: {} x: {} y: {}".format(num_steps, round(previous,2), funcOne(previous)))
	return funcOne(previous)

######################### Exp ################################
# prevents printing in scientific notation
np.set_printoptions(suppress=True)

pop = initPopFunc1(100)
numGen = 25
mutationRate = 0.1

"""
# Function 1 GA

res = []
fittness = []
for i in range(50):
	best = -9999
	for generation in range(1,numGen + 1):
		#return population sorted by fittness value
		df = calcFitness(pop)
		cur, avgFitness = stats(df,generation)
		if cur > best:
			best = cur
			fittness.append(avgFitness)
		# select top 50% fittest individuals
		parents = df[50: 100, :]
		parents = parents[:,0]
		group1 = replicate(parents, mutationRate)
		group2 = replicate(parents, mutationRate)
		group1.extend(group2)
		# new generation
		pop = np.array(group1)
		if best >= 74.9:
			break
	res.append(best)
"""

"""
# Function 2 GA
bestValues = []
avgFitness = []

pop = initPopFunc2(100)
numGen = 50
df = calcFitness2(pop)

for i in range(50):
	pop = initPopFunc2(100)
	for generation in range(1, numGen + 1):
		df = calcFitness2(pop)
		cur = stats2(df,generation)
		parents = df[50: 100, :]
		parents = parents[:,[0,1]]
		group1 = replicate2(parents, mutationRate)
		group2 = replicate2(parents, mutationRate)
		pop = np.vstack((group1, group2))
	bestV, avgFit = stats2(df,25)
	bestValues.append(bestV)
	avgFitness.append(avgFit)

genHistFunc2(bestValues, "f(x,y)", "Best Value Found function 2: 50 trials")
genHillGraph(bestValues)
"""

"""
# Exp: hill climber function 1
best = []
for i in range(100):
	best_val = hillClimbFunc1(0.01)
	best.append(round(best_val,2))
	print("Optima = {}".format(best_val))

genHillGraph(best)
"""

"""
# Exp: hill climber function 2
best = []
for i in range(1000):
	best_val, x, y = hillClimbFunc2(0.01)
	best.append(round(best_val,2))
	print("x = {} y = {} Optima = {}".format(x,y, best_val))

genHillGraph2(best)
"""




