import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import numpy as np
import random

######################### Stats ################################

# gen stats about population function 1
def stats(pop, iter):
	fitness = pop[:,2]
	Y = pop[:,1]
	fittestRow = pop[len(pop) - 1]
	weakestRow = pop[0]
	bestX = fittestRow[0]
	bestY = fittestRow[1]
	worstX = weakestRow[0]
	worstY = weakestRow[1]
	avgFitness = np.mean(fitness)
	avgValue = np.mean(Y)
	printStats(iter,avgFitness, avgValue, funcOne(bestX), funcOne(worstX))
	return bestY, avgFitness

# gen stats about population function 2
def stats2(pop, iter):
	fitness = pop[:,3]
	z = pop[:,2]
	fittestRow = pop[len(pop) - 1]
	weakestRow = pop[0]
	bestX = fittestRow[0]
	bestY = fittestRow[1]
	bestZ = fittestRow[2]
	worstX = weakestRow[0]
	worstY = weakestRow[1]
	worstZ = weakestRow[2]
	avgFitness = np.mean(fitness)
	avgValue = np.mean(z)
	printStats(iter,avgFitness, avgValue, funcTwo(bestX, bestY), funcTwo(worstX, worstY))
	return funcTwo(bestX, bestY), avgFitness

def printStats(gen, avgFitness, avgOutput, bestOutput, worstOutput):
	print("\tGeneration: {}".format(gen))
	print("Average Fitness: {}\nAverage Output: {}".format(round(avgFitness,4),round(avgOutput,4)))
	print("Best: {}".format(round(bestOutput,2)))
	print("Worst:{}\n".format(round(worstOutput,2)))

######################### Graphs ################################

def genHillGraph2(optima):
	labels = ["Global", "Local"]
	found = 0
	local = 0
	for i in optima:
		if i >= 140:
			found = found + 1
		else:
			local = local +  1
	sizes = [found, local]
	fig1, ax1 = plt.subplots()
	ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
	        shadow=True, startangle=90)
	ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
	plt.title("Global Optima / Local Optima  Iter = 50")
	plt.show()

def genHistFunc2(data, xlable, titleLabel):
	plt.hist(data)
	plt.xlabel(xlable)
	plt.title(titleLabel)
	plt.show()

def genHillGraph(best):
	found = 0
	local = 0
	for i in best:
		if i > 74:
			found = found + 1
		else:
			local = local + 1
	labels = ["Global", "Local"]
	sizes = [(found), local]
	fig1, ax1 = plt.subplots()
	ax1.pie(sizes, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
	ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
	plt.title("Global Optima / Local Optima")
	plt.show()

def funcOne(x):
	return round(x**4 - (22 * x**2),4)

def funcTwo(x,y):
	return round(x**4 - (22 * x**2) + (y**4) - (22 * y**2), 4)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

######################### Misc ################################

def initPopFunc1(initPop):
	return np.random.uniform(-5,5, initPop)

def initPopFunc2(initPop):
	X = np.random.uniform(-5,5, initPop)
	Y = np.random.uniform(-5,5, initPop)
	res = np.column_stack((X,Y))
	return res

def findOptimum():
	maxVal = -999999
	data = [-5,-4,-3,-2,-1,0,1,2,3,4,5]
	for x in data:
		for y in data:
			res = funcTwo(x,y)
			print(res)
			if res >= maxVal:
				maxVal = res
				print("({},{}) = {}".format(x,y,maxVal))


