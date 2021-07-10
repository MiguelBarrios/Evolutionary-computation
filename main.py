import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt

def gaussianMutationScale(x,y,sd, scale):
	xprime = x + (scale * (np.random.normal(0,sd)))
	yprime = y + (scale * np.random.normal(0,sd))
	return xprime, yprime

def additiveMutationGlobal(x, y):
	xprime = np.random.uniform(-100,100, len(x))
	yprime = np.random.uniform(-100,100, len(x))
	return xprime, yprime


scaleFactor = 0.5
sd = 1
generation = 0
numGenerations = 1 + 1
palette = sns.color_palette(None, numGenerations + 1)

# initial population
x = np.random.uniform(low = -100, high = 100, size = 100)
y = np.random.uniform(low = -100, high = 100, size = 100)

"""
# Local Search
plt.title("Local Search P({})".format(numGenerations - 1))
plt.scatter(x,y,marker = '.', c = palette[0])
for i in range(numGenerations - 1):
	if i % 50 == 0:
		scaleFactor = scaleFactor + (scaleFactor * 0.05)
	x1,y1 = gaussianMutationScale(x,y, sd, scaleFactor)
	plt.scatter(x1,y1,marker = '.', c = palette[i + 1])
plt.show()
"""

"""
# Global Search
plt.title("Global Search P({})".format(numGenerations - 1))
plt.scatter(x,y,marker = '.', c = palette[0])
for i in range(numGenerations - 1):
	x1,y1 = additiveMutationGlobal(x, y)
	plt.scatter(x1,y1,marker = '.', c = palette[i + 1])
plt.show()
"""

"""
# Combined Search
S = 2
arr = np.vstack((x,y)).T
plt.title("Combined Search P({})".format(numGenerations - 1))
plt.scatter(x,y,marker = '.', c = palette[0], s = S)
for i in range(numGenerations - 1):
	if i % 50 == 0:
		scaleFactor = scaleFactor + (scaleFactor * 0.05)
	arr2 = np.random.shuffle(arr)
	split = np.vsplit(arr,2)
	one = split[0]
	two = split[1]
	x1 = one[:,0]
	y1 = one[:,1]
	x2 = two[:,0]
	y2 = two[:,1]
	x1,y1 = gaussianMutationScale(x1, y1,sd, scaleFactor)
	x2,y2 = additiveMutationGlobal(x2, y2)
	plt.scatter(x1,y1,marker = '.', c = 'red', s = S)
	plt.scatter(x2,y2,marker = '.', c = 'blue', s = S)
plt.show()
"""
