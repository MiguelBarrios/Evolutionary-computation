import numpy as np 
import matplotlib.pyplot as plt 

def graphProblem2(x,y):
	# Gen problem 2 plot
	fig = plt.figure()
	ax = fig.add_subplot()
	for xy in zip(x, y):                                       # <--
	    ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data') # <--
	plt.scatter(x,y, marker = ".")
	plt.xlabel("x")
	plt.ylabel("y")
	plt.show()

def graphProblem3(weights):
	for i in range(len(IDs)):
		slope, y_intercept = getLine(weights[i])
		y1 = slope * (-5) + y_intercept
		y2 = slope * (5) + y_intercept
		plt.plot([-5,5], [y1,y2], label = IDs[i])
	for xy in zip(x, y):                                       # <--
	    plt.annotate('(%s, %s)' % xy, xy=xy, textcoords='data') # <--
	plt.scatter(x,y,color = 'Black')
	plt.legend(loc = 'upper left', bbox_to_anchor=(1, 1))
	plt.show()

def objective_fitness(w, ID):
	correct = 0
	for i in range(len(x)):
		net1, res = net(x[i],y[i], w)
		o = output[i]
		#print("\texpected = {} actual = {}".format(o, res))
		if(res == o):
			correct = correct + 1
	print("Weights {}: v1 = {} v2 = {} v3 = {} Fitness {} : T Fitness {}\n".format(ID,w[0],w[1],w[2],correct, correct - (len(x) - correct)))
	return correct

def net(x1,x2, weights):
	net = x1 * weights[0] + x2 * weights[1] + (-1) * weights[2]
	if net >= 0:
		out = 1
	else:
		out = 0
	#print("\tinput: ({},{}) : ".format(x1,x2), end = " ")
	#print("net = {} out = {}".format(net,out))
	return net,out

def addPointsToPlot(x,y, labeled):
	fig = plt.figure()
	if(labeled):
		ax = fig.add_subplot()
		for xy in zip(x, y):                                      
		    ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')
	plt.scatter(x,y, marker = ".")

def addLineToPlot(weights, range, ID):
	slope, y_intercept = getLine(weights)
	y1 = slope * (-range) + y_intercept
	y2 = slope * (range) + y_intercept
	plt.plot([-range,range], [y1,y2], label = ID)

def getLine(weights):
	v1 = weights[0]
	v2 = weights[1] * -1
	v3 = weights[2] * -1
	slope = round(v1 / v2,2)
	y_intercept = round(v3/v2,2)
	return slope, y_intercept

y1 = 1
y2 = 0

x = np.array([-0.1,0.0,0.2,1.3,-1.2,-0.5])
y = np.array([-0.8,0.0,-0.2,1.3,-0.4,0.2])
output = np.array([y1,y2,y2,y2,y1,y1])
z = -1

IDs = ['A','B','C','D','E','F','G','H','I','J']
weights = np.array([[-1.0,-0.7,0.0],
					[-0.9,0.9,0.3],
					[0.9,-0.3,-0.9],
					[-0.1,0.8,-0.4],
					[1.0,0.5,-0.5],
					[-0.8,0.1,0.6],
					[-0.1,0.8,0.3],
					[0.6,0.8,0.2],
					[-0.3,-0.2,1.0],
					[0.6,0.4,0.3]])

"""
Problem 5
j = 0
correct = 0
for w in weights:
	print("Weights {}: v1 = {} v2 = {} v3 = {}".format(IDs[j],w[0],w[1],w[2]))
	for i in range(len(x)):
		res = net(x[i],y[i], w)
		o = output[i]
		if(res == o):
			correct = correct + 1
	print("Fitness {} : T Fitness {}\n".format(correct, correct - (len(x) - correct)))
	correct = 0
	j = j + 1

"""

# Problem 10
"""
A = weights[0]
B = weights[1]

K = np.array([A[0], B[1], B[2]])
L = np.array([B[0], A[1], A[2]])
fig = plt.figure()

addPointsToPlot(x,y, labeled = True)
addLineToPlot(K, plt, 2, "K")
addLineToPlot(A, plt, 2, "A")
addLineToPlot(L, plt, 2, "L")

plt.legend(loc = 'lower right')
plt.xlabel("x")
plt.ylabel("y")
plt.show()

objective_fitness(K,"K")
#objective_fitness(L,"L")
objective_fitness(A, "A")
"""



"""
# Problem 12
print("Fitness befor new Points")
fitness = 0
for i in range(len(weights)):
	fitness = fitness + objective_fitness(weights[i], IDs[i])

print("Avg Fitness {}".format(fitness / len(x)))
"""










