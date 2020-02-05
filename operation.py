import numpy as np
import math as math 
import sympy as sympy


class Operation: 
	def __init__(self,opts):
		for k,v in opts.items():
			setattr(self,k,v)
	def get_op(self):
		return self.f


# Linear operation: Matrix/vector, matrix/matrix multiplication 

def matmulF(inputs,layer):
	if hasattr(layer,'weights'):
		result = inputs.dot(layer.weights)
		if hasattr(layer,'bias'):
			result += layer.bias
	else: 
		result = inputs
	return result

def matmulB(inputs, layer, grad, is_cost=None,is_weights=None, is_bias=None):
	if is_cost:
	# First, compute gradient on the layer inputs (inputs passed to the layer from the previous layer in forward graph)
	   gradient = inputs * grad
	elif is_weights:
		gradient = inputs.T.dot(grad)
	elif is_bias:
		grad = grad.sum(axis=0)
		return grad 
	else:
		gradient = inputs * grad # should be gadamard product
    
    # # For bias

    # gradient = gradient + inputs.bias

	return gradient 

# Define Operation 

matmul = Operation({
	'name': 'linear',
	'f': matmulF,
	'bprop': matmulB
	})

# Rectified Linear Unit
def reluTest(inputs):
	print(inputs.dtype,'Inputs type is ...')
	def _f(inputs):
		if inputs > 0:
			return inputs
		else: 
			return 0.1
	vectorized = np.vectorize(_f)
	result = vectorized(inputs)
	print (result, 'Relu result')
	return result


def reluF(inputs,layer):
	def _f(inputs):
		if inputs > 0:
			return inputs
		else: 
			return 0
	vectorized = np.vectorize(_f)
	if hasattr(layer,'weights'):
		result = vectorized(inputs).dot(layer.weights)
	else: 
		result = vectorized(inputs)
	print (result, 'Relu result')
	return result

def reluB(inputs, layer, grad,is_cost=0,is_parameter=0,dLoss_dRelu=0, dRelu_dInput=0,bias=0):
	# Relu derivative is one for x>0 and 0 for
	def _derivative (value):  
		if value > 0:
			return 1
		else:
			return 0

	vectorDiff = np.vectorize(_derivative, otypes=[np.float])
	copy = inputs.copy() 
	reluDrvt = vectorDiff(copy)
	if is_parameter:
		gradient = reluDrvt.T.dot(grad)
	elif dLoss_dRelu:
		gradient = grad.dot(layer.weights.T)
	elif dRelu_dInput:
		# g_copy = grad.copy().astype('float64') 
		# print (g_copy, 'grad copy ')
		gradient =  grad * reluDrvt 
	elif bias:
		return reluDrvt
	else: 
		gradient = grad * reluDrvt
	return gradient 

relu = Operation({
	'name': 'relu',
	'f': reluF,
	'bprop': reluB
	}) 


Operations = {
	'linear': matmul,
	'relu':  relu 
}












