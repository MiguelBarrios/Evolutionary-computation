import numpy as np 
import scipy.stats
import random
import seaborn as sns
import matplotlib.pyplot as plt

def initPop(type, N):
	if type == "uniform":
		return np.random.uniform(-100,100, N)
	mu = 0
	sigma = 20
	lower = -100
	upper = 100
	if type == "gaussian":
		return scipy.stats.truncnorm.rvs(
          (lower-mu)/sigma,(upper-mu)/sigma,loc=mu,scale=sigma,size=N)


random.seed(360)

X = initPop("gaussian", 100)
Y = initPop("gaussian", 100)
plt.scatter(X,Y, marker = '.')
chromosomes = np.vstack((X,Y)).T

#Shuffle 
maxGenerations = 10
row = 0
palette = sns.color_palette(None, maxGenerations + 1)
for generation in range(maxGenerations):
	# Shuffle so each pair of parents is random
	np.random.shuffle(chromosomes)
	row = 0
	while row < len(chromosomes):
		parent1 = chromosomes[row,:]
		parent2 = chromosomes[row + 1,:]
		x1 = parent1[0]
		y1 = parent1[1]
		x2 = parent2[0]
		y2 = parent2[1]
		# CrossOver
		childOne = np.array([x1,y2])
		childTwo = np.array([x2,y1])
		chromosomes[row] = childOne
		chromosomes[row + 1] = childTwo
		row = row + 2
	print("\n\n")
	x = chromosomes[:,0]
	y = chromosomes[:,1]
	plt.scatter(x,y,marker = '.', c = palette[generation])
	
plt.show()



