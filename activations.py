import numpy as np 
import math as math 
from scipy import special

# def sigmoid(x):
# 	def _s(x):
# 		sigmoid = math.exp(-np.logaddexp(0, -x))
# 		return sigmoid 
# 	vectorized = np.vectorize(_s)
# 	return vectorized(x) 

def sigmoid(x):
	return special.expit(x)

def sigmoid_dt(x):
	dt = sigmoid(x) * (1 - sigmoid(x))
	return dt 

def softmax (predictions):
	return special.softmax(predictions)
	# normalized = predictions - np.max(predictions)
	# out = np.exp(normalized)
	# return out / np.sum(out)

def softmax_dt (output,input):
	# Initialize hessian 
	hessian = np.zeros((len(output),len(input)))
	for i in range(len(output)):
		for j in range(len(input)):
			if j == i:
				hessian [i,j] = output[i] * (1 - output[i])
			else: 
				hessian [i,j] = - output[i] * output[j]
	return hessian 

def test_softmax_dt ():
	input = np.random.randint(3000,3500,(10)).reshape(10,1)
	output = softmax(input)
	hessian = softmax_dt(output, input)