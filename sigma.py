import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0,0],[0,1],[1,0],[1,1]],dtype='float64')
Y = np.array([[0],[1],[1],[0]], dtype='float64')
W1 = np.random.rand(2,2)
W2 = np.random.rand(2,1)
b1 = np.random.rand(1,2)


alpha = 0.01

cache = {'W1': W1,'W2': W2,'b1': b1, 'l1':None,'s1': None,'l2': None,'s2': None}

def sigmoid(x):
	def _s(x):
		sigmoid = 1 / (1 + np.exp(-x))
		return sigmoid 
	vectorized = np.vectorize(_s)
	return vectorized(x) 

def sigmoid_dt(x):
	dt = sigmoid(x) * (1 - sigmoid(x))
	return dt
def matmul(input, weights ):
	product = input.dot(weights)
	return product 
def calc_cost (output,target):
	cost = np.sum(1/2 * (output - target)**2)
	print (cost, 'Training cost')
	return cost 
def cost_dt (output,target):
	return output - target

def forwardprop (cache):
	cache['l1'] = matmul(X,cache['W1']) + cache ['b1']
	cache['s1'] = sigmoid (cache['l1'])
	cache['l2'] = matmul(cache['s1'],cache['W2'])
	cache['s2'] = sigmoid(cache['l2'])

def backprop (cache):
	dtCost = cost_dt(cache['s2'],Y)
	dtS2 = dtCost * sigmoid_dt(cache['s2'])
	dtW2 = cache['s1'].T.dot(dtS2)
	dtL2 = dtS2.dot(cache['W2'].T)
	dtS1 = dtL2 * sigmoid_dt(cache['l1'])
	dtW1 = X.T.dot(dtS1)
	dtb1 = np.sum(dtS1)

	# Weights update

	cache['W2'] -= alpha * dtW2
	cache ['W1'] -= alpha * dtW1
	cache ['b1'] -= alpha * dtb1

	print (cache['W2'], 'W2 weights after update')
	print (cache['W1'], 'W1 weights after update')
	print (cache['b1'], 'Layer 1 bias after update')


def train ():
	history = []
	iterations = 1000000
	cost = 1000
	for i in range(iterations):
		forwardprop (cache)
		cost = calc_cost (cache['s2'],Y)
		backprop(cache)
		history.append(cost)

	if cost < 1:
		print ('Model converged')
		print(cache['s2'],'Output')
		print(cache['W2'],'W2 weights')
		print(cache['W1'],'W1 weights')
		print(cache['b1'],'bias weights')
		x = []
		y = []
		for i,val in enumerate(history):
			x.append(i)
			y.append(val) 
		plt.plot(x,y)
		plt.xlabel('Iterations')
		plt.ylabel('Lost')
		plt.title("XOR training with sigmoid hidden layer")
		plt.legend()
		plt.show()






	