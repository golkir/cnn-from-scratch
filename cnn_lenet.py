import numpy as np
import copy
from sklearn.metrics import log_loss  
import lib as lib
from maxpool import *
import matplotlib.pyplot as plt
from scipy import signal
from mlxtend.data import loadlocal_mnist
from keras.datasets import mnist
import activations as activations 
import losses as losses

X, y = loadlocal_mnist(
        images_path='/Users/kirillgoltsman/Documents/AI-ML/CNN_backprop/mnist/train-images-idx3-ubyte', 
        labels_path='/Users/kirillgoltsman/Documents/AI-ML/CNN_backprop/mnist/train-labels-idx1-ubyte',

        )
test_set, test_labels = loadlocal_mnist(
	                   images_path = '/Users/kirillgoltsman/Documents/AI-ML/CNN_backprop/mnist/t10k-images-idx3-ubyte',
	                   labels_path = '/Users/kirillgoltsman/Documents/AI-ML/CNN_backprop/mnist/t10k-labels-idx1-ubyte'
	                   )

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Pre-process data

X = X.reshape((-1,28,28))

X = np.pad(X, [(0,0),(1,1), (1,1)], 'constant')

tr_set, validation_set =  X[:50000], X[50000:]

tr_labels, validation_labels = y[:50000], y[50000:]

tr_labels = lib.to_onehot(tr_labels)

validation_labels = lib.to_onehot(validation_labels)

test_set = test_set.reshape((-1,28,28))
test_set = np.pad(test_set, [(0,0),(1,1), (1,1)], 'constant')
test_labels = lib.to_onehot(test_labels)

# Keras Mnist dataset 
x_train = np.pad(x_train, [(0,0),(1,1), (1,1)], 'constant')
y_train = lib.to_onehot(y_train)
x_test = np.pad(x_test, [(0,0),(1,1), (1,1)], 'constant')
y_test = lib.to_onehot(y_test)


def forwardprop(image,cache):
	
	"""
	 C1 Convolution Layer
	 Input: 30X30
	 Filters: 8
	 Kernel: 3X3
	 Output: 28X28

	"""
	cache['C1']['fmaps'] = []

	for index in range(cache['C1']['filters'].shape[0]):
		cache['C1']['fmaps'].append(signal.correlate(image, cache['C1']['filters'][index], mode='valid'))
	
	cache['C1']['fmaps'] = np.asarray(cache['C1']['fmaps']) + cache['C1']['bias']

	# print(cache['C1']['fmaps'].shape, 'C1 fmaps')


	""" 
	Maxpooling
	Input: 8X28X28
	Maxpool kernel: 2X2
	Output: 8X14X14

	"""

	cache['C1']['maxpool'] = maxpool_run(cache['C1']['fmaps'],(2,2))


	# print(cache['C1']['maxpool'].shape, 'C1 Maxpool' )


	"""
	Sigmoid Activation
	Input: 8X14X14
	Output: 8X14X14

	"""

	cache['C1']['sigmoid'] = activations.sigmoid(cache['C1']['maxpool'])

	# print(cache['C1']['sigmoid'].shape, 'C1 sigmoid')

	"""
	C2 Convolution Layer
	Input: 8X14X14
	Filters: 16
	Kernel: 5X5
	Output: 16X10X10

	"""

	cache['C2']['fmaps'] = []

	for i in range(cache['C2']['filters'].shape[0]):
		fmap = np.sum(signal.correlate(cache['C1']['sigmoid'], cache['C2']['filters'][i].reshape(1,5,5), mode='valid', method='direct'), axis=0)
		cache['C2']['fmaps'].append(fmap)
	
	cache['C2']['fmaps'] = np.asarray(cache['C2']['fmaps']) + cache['C2']['bias']

	# print(cache['C2']['fmaps'].shape, 'C2 fmaps')


	"""
	Maxpooling 
	Input: 16X10X10
	Kernel: 2X2
	Output: 16X5X5
	"""

	cache['C2']['maxpool'] = maxpool_run(cache['C2']['fmaps'],(2,2))

	# print (cache['C2']['maxpool'].shape , 'C2 maxpool')

	"""
	Sigmoid Activation
	Input: 16X5X5
	Output: 16X5X5
	"""

	cache['C2']['sigmoid'] = activations.sigmoid(cache['C2']['maxpool'])

	# print(cache['C2']['sigmoid'].shape, 'C2 sigmoid')

	"""
	C3 Dense Layer (fully connected)
	Input: 16X5X5
	Flatten: 400X1
	Weights: 120X400
	Output: 120X1 
	"""

	flatten = cache['C2']['sigmoid'].flatten().reshape(400,1)

	cache['C3']['fmaps'] = cache['C3']['filters'].dot(flatten) + cache['C3']['bias']
	
	# print(cache['C3']['fmaps'].shape , 'C3 fmaps')

	"""
	Sigmoid Activation
	Input: 120X1
	Output: 120X1
	"""

	cache['C3']['sigmoid'] = activations.sigmoid(cache['C3']['fmaps'])

	# print(cache['C3']['sigmoid'].shape, 'C3 sigmoid shape')
	
	"""
	F6: Dense layer (fully connected) 
	Input: 120X1
	Weights: 84X120
	Output: 84X1 

	"""

	cache['F6']['fmaps'] = cache['F6']['filters'].dot(cache['C3']['sigmoid']) + cache['F6']['bias']

	# print(cache ['F6']['fmaps'].shape, 'F6 fmaps')

	"""
	Sigmoid activation
	Input: 84X1
	Output: 84X1
	"""

	cache['F6']['sigmoid'] = activations.sigmoid(cache['F6']['fmaps'])

	"""
	F7: Dense Layer (fully connected)
	Input: 84X1
	Weights: 10X84
	Output: 1X10
	"""

	cache['F7']['fmaps'] = cache['F7']['filters'].dot(cache['F6']['sigmoid']) + cache['F7']['bias']

	"""
	Softmax Activation
	Output: 1X10

	"""
	cache['softmax'] = activations.softmax(cache['F7']['fmaps']).T

	# print(cache['softmax'].shape, 'Softmax result')

	return cache


def backprop(cache, example, label):


	"""
	Gradient of loss w.r.t to softmax

	"""

	dL_dSoftmax = cache['softmax'] - label
	
	# print (dL_dSoftmax.shape, 'Shape of gradient of loss w.r.t to softmax')

	"""
	Gradient of loss w.r.t to F7 parameters
	Dimensions:  (84X1 .dot 1X10).T = 10X84
	"""

	dL_dF7_parameters = cache['F6']['sigmoid'].dot(dL_dSoftmax).T

	"""
	F7 Biases Gradient
	Output: 10X1
	"""
	dL_dF7_biases = dL_dSoftmax.T

	"""
	Gradient of loss with respect to F7 layer inputs
	Output: (1X10 .dot 10X84 ).T = 84X1
	"""

	dL_dF7 = dL_dSoftmax.dot(cache['F7']['filters']).T

	"""
	Local derrivative of F6 sigmoid
	Output: 84X1
	"""

	dF6_sigmoid = activations.sigmoid_dt(cache['F6']['fmaps'])

	"""
	Gradient of loss w.r.t to F6 sigmoid
	Output: 84X1 * 84X1 = 84X1 (Gadamard product)

	"""

	dL_dF6_sigmoid = dL_dF7 * dF6_sigmoid

	"""
	Gradient of loss w.r.t F6 parameters
	Output: 84X1 .dot 120X1.T = 84X120 
	"""

	dL_dF6_parameters = dL_dF6_sigmoid.dot(cache['C3']['sigmoid'].T)

	# print (dL_dF6_parameters.shape, 'Gradient of loss with respect to F7 dense layer parameters')

	"""
	Gradient of F6 biases
	"""

	dL_dF6_biases = dL_dF6_sigmoid

	# print(dL_dF6_biases.shape, 'Gradient of loss w.r.t F7 biases')

	"""
	Gradient of loss w.r.t F6 inputs 
	Output: 84X120.T .dot 84X1  = 120X1
	"""

	dL_dF6 = cache['F6']['filters'].T.dot(dL_dF6_sigmoid)

	# print (dL_dF6.shape, 'Grad of loss w.r.t to F7 inputs')


	"""
	C3 Sigmoid Local Derrivative
	Output: 120X1
	"""

	dC3_sigmoid = activations.sigmoid_dt(cache['C3']['fmaps'])

	# print (dC3_sigmoid.shape, 'C3 Sigmoid derrivative')

	"""
	Gradient of loss with respect to C3 sigmoid 
	Output: 120 X 1  * 120 X 1 = 120 X 1 ( Gadamard product)

	"""
	dL_dC3_sigmoid = dL_dF6 * dC3_sigmoid

	# print(dL_dC3_sigmoid.shape, 'Gradient of loss w.r.t C3 sigmoid')

	"""
	Gradient of loss with respect to C3 parameters 
	Output: 120X1 .dot 1X400 = 120X400
	"""
	dL_dC3_parameters = dL_dC3_sigmoid.dot(cache['C2']['sigmoid'].flatten().reshape(1,400))

	# print (dL_dC3_parameters.shape, 'C3 parameters gradient' )


	"""
	Gradient of C3 biases
	"""

	dL_dC3_biases = dL_dC3_sigmoid

	# print(dL_dC3_biases.shape, 'C3 biases')

	
	"""
	Gradient of loss with respect to C3 Layer Inputs
	Output: 120X1.T .dot 120X400 = 1X400 = 16X5X5

	"""

	dL_dC3 = dL_dC3_sigmoid.T.dot(cache['C3']['filters']).reshape(16,5,5)

	# print (dL_dC3.shape, 'Gradient of loss w.r.t to C3 inputs')

	"""
	Local derrivative of C2 sigmoid
	Output: 16X5X5

	"""
	dC2_sigmoid = activations.sigmoid_dt(cache['C2']['maxpool'])

	# print(dC2_sigmoid.shape, 'Gradient of C2 sigmoid')

	"""
	Gradient of loss w.r.t to C2 Sigmoid
	Output: 16X5X5 * 16X5X5 = 16X5X5 (Gadamard product)
	"""

	dL_dC2_sigmoid = dL_dC3 * dC2_sigmoid

	# print(dL_dC2_sigmoid.shape, 'Gradient of loss w.r.t to C2 sigmoid')

	"""
	Gradient of loss with respect to C2 Maxpool
	Output: 16X10X10

	"""

	dL_dS2_maxpool = []
	for sigmoid in range(dL_dC2_sigmoid.shape[0]):
		dt = maxpool_backprop(cache['C2']['fmaps'][sigmoid], (2,2), dL_dC2_sigmoid[sigmoid])
		dL_dS2_maxpool.append(dt)
	dL_dS2_maxpool = np.asarray(dL_dS2_maxpool)


	# print (dL_dS2_maxpool.shape, 'Gradient of loss w.r.t to S2 subsampling layer')


	"""
	Gradient of loss w.r.t C2 convolution layer filters
	Output: 16X10X10 convolve over 16X14X14 = 16X5X5

	"""
	dL_dC2_parameters = []

	for i in range(dL_dS2_maxpool.shape[0]):
		gr = np.sum(signal.correlate(cache['C1']['sigmoid'], dL_dS2_maxpool[i].reshape(1,10,10), mode='valid', method='direct'), axis=0)
		dL_dC2_parameters.append(gr)
	
	dL_dC2_parameters = np.asarray(dL_dC2_parameters)
	
	# print (dL_dC2_parameters.shape, 'Gradient of loss w.r.t C2 filters/parameters')


	"""
	Gradient of loss w.r.t C2 Bias
	Output: 16X1X1
	"""

	dL_dC2_biases = np.array([ np.sum(dL_dS2_maxpool[i]) for i in range(dL_dS2_maxpool.shape[0]) ]).reshape(16,1,1)


	# print (dL_dC2_biases.shape, 'Gradient of loss w.r.t to C2 bias')


	"""
	Gradient of loss w.r.t C2 inputs
	Operation:  "full convolution" of C2 Maxpool with C2 filters 
	Output: sum (16X10X10 full-convolve over 16X5X5) = 8X14X14 (same gradient for all 8 feature maps from C1)

	"""
	fconvs = []
	for g in range(dL_dS2_maxpool.shape[0]):
		fconv = signal.convolve2d(dL_dS2_maxpool[g], cache['C2']['filters'][g], mode='full')
		fconvs.append(fconv)
	fconvs = np.sum(np.asarray(fconvs),axis=0)

	# The same gradient for all C2 inputs
	dL_dC2 = np.asarray([fconvs for i in range(8)])

	# print(dL_dC2.shape, 'Gradient of loss w.r.t to C2 inputs')

	"""
	Local derrivative of C1 sigmoid
	Output: 8X14X14

	"""
	dC1_sigmoid = activations.sigmoid_dt(cache['C1']['maxpool'])

	# print (dC1_sigmoid.shape, 'Sigmoid derrivative')

	"""
	Gradient of loss w.r.t to S1 sigmoid
	Output: 8X14X14 * 8X14X14 = 8X14X14 (Gadamard product)
	"""

	dL_dC1_sigmoid = dL_dC2 * dC1_sigmoid

	# print(dL_dC1_sigmoid.shape, 'Gradient of loss w.r.t to C1 sigmoid')

	"""
	C1 Maxpooling 
	Output: 8X28X28

	"""

	dL_dC1_maxpool = np.asarray([ maxpool_backprop(cache['C1']['fmaps'][sigmoid_i], (2,2), sigmoid) for sigmoid_i, sigmoid in enumerate(dL_dC1_sigmoid) ])

	# print (dL_dC1_maxpool.shape, 'Gradient of loss w.r.t to C1 S1 subsampling layer')


	"""
	Gradient of loss w.r.t to C1 filters
	Output: 8X28X28 over 1X32X32 (image) = 8X5X5

	"""

	dL_dC1_parameters = np.asarray([signal.correlate(example, dL_dC1_maxpool[i], mode='valid', method='direct') for i in range(0,dL_dC1_maxpool.shape[0])])

	# print (dL_dC1_parameters.shape, 'Gradient of loss w.r.t to C1 filters')

	"""
	Gradient of loss w.r.t C1 bias
	"""

	dL_dC1_biases = np.array([ np.sum(dL_dC1_maxpool[i]) for i in range(dL_dC1_maxpool.shape[0]) ]).reshape(8,1,1)

	# print (dL_dC1_biases.shape, 'Gradient of loss w.r.t to C1 biases')

	return { 'parameters': np.array([dL_dC1_parameters, dL_dC2_parameters, dL_dC3_parameters, dL_dF6_parameters, dL_dF7_parameters]),
	         'biases': np.array([dL_dC1_biases, dL_dC2_biases, dL_dC3_biases,dL_dF6_biases, dL_dF7_biases])}


def update_weights (grads_params, grads_biases,cache,lr):

	cache['C1']['filters'] -= lr * grads_params[0]
	cache ['C2'] ['filters'] -= lr * grads_params [1]
	cache['C3']['filters'] -= lr * grads_params[2]
	cache['F6']['filters'] -= lr * grads_params [3]
	cache['F7']['filters'] -= lr * grads_params[4]
	cache ['C1']['bias'] -= lr * grads_biases[0]
	cache['C2']['bias'] -= lr * grads_biases[1]
	cache['C3']['bias'] -= lr * grads_biases[2]
	cache ['F6']['bias'] -= lr * grads_biases[3]
	cache['F7']['bias'] -= lr * grads_biases[4]

	return cache 


def predict(model,dataset, labels):
	prediction_history = []
	accuracy = 0
	for index in range(dataset.shape[0]):
		image = lib.image_normalize(dataset[index])
		fprop = forwardprop(image, model)
		cost = log_loss(labels[index].reshape(1,10), fprop['softmax'])
		prediction_history.append(cost)
		if cost < 0.01:
			accuracy += 1
			print('the prediction is accurate')
		else:
			print('the prediction is wrong')
		print ('the predicted value is:', model['softmax'])
		print ('the actual value is:', labels[index])
	model_accuracy = accuracy / len (dataset) * 100 

	print ('Model accuracy is:', model_accuracy, '%')
	lib.visualizeCost(prediction_history)

def gradient_check_test(cache, X, Y, gradients, parameters, layer_name, parameter_type, epsilon=1e-7):
	grad_approx = []
	for i in range(len(parameters.flatten())):
		thetaplus = parameters.copy().flatten()
		thetaplus[i] = thetaplus[i] + epsilon
		cache_copy = cache.copy()
		cache_copy[layer_name][parameter_type] = thetaplus.reshape(parameters.shape)
		cache_copy = forwardprop(X, cache_copy)
		loss_thetaplus = log_loss(Y.reshape(1,10), cache_copy['softmax'])
		thetaminus = parameters.copy().flatten()
		thetaminus[i] = thetaminus[i] - epsilon
		cache_copy_2 = cache.copy()
		cache_copy_2[layer_name][parameter_type] = thetaminus.reshape(parameters.shape)
		cache_copy_2 = forwardprop(X, cache_copy_2)
		loss_thetaminus = log_loss(Y.reshape(1,10), cache_copy_2['softmax'])
		dt = (loss_thetaplus  - loss_thetaminus) / (epsilon * 2)
		grad_approx.append(dt)
	grad_approx = np.array(grad_approx).reshape(parameters.shape)
	# print(grad_approx,'Approximated grad')
	# print (gradients, 'Computed gradient')
	numerator = np.linalg.norm(gradients - grad_approx)                                    
	denominator = np.linalg.norm(gradients) + np.linalg.norm(grad_approx)                
	difference = numerator / denominator  
	if difference > 1e-7:
		print("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
	else:
		print("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")
	return difference

def sgd_online(training_set, labels, epochs):
	N = training_set.shape[0]
	weighted_avg_history = [{'parameters':0, 'biases': 0}]
	cost_history = []
	lr = 0.3
	cache = { 'C1': {'fmaps': None, 'bias': np.zeros((8,1,1)), 'filters': lib.init_weights_glorot((8,3,3), 784, 6272), 'maxpool': None, 'sigmoid': None },
	          'C2': {'fmaps': None, 'filters': lib.init_weights_glorot((16,5,5), 1568, 1600), 'bias': np.zeros((16,1,1)), 'maxpool': None, 'sigmoid': None },
	          'C3': {'fmaps': None, 'filters': lib.init_weights_glorot((120,400), 400, 120), 'bias': np.zeros((120,1)), 'sigmoid': None},
	          'F6': {'fmaps': None, 'filters': lib.init_weights_glorot((84,120), 120, 84), 'bias': np.zeros((84,1)), 'sigmoid': None},
	          'F7': {'fmaps': None, 'filters': lib.init_weights_glorot((10,84),84,10), 'bias': np.zeros((10,1))},
	          'softmax': None
	}

	for epoch in range(epochs):
		permutation = np.random.permutation(N) # Shuffle training set 

		X, Y = training_set[permutation], labels[permutation]

		for image in range(X.shape[0]):
			X[image] = lib.image_normalize(X[image])

		for ex in range(X.shape[0]):

			# if ex in [4000,8000,10000,12000,18000]:
			# 	lib.visualizeCost(cost_history)

			print ('Epoch:', epoch, 'Example:', ex)
			print (lr, 'Current learning rate')
			label = Y[ex]
			cache_copy = forwardprop(X[ex], copy.deepcopy(cache))
			grad = backprop(cache_copy, X[ex], label)
			cost = log_loss(label.reshape(1,10), cache_copy['softmax'])
			
			gradient_check_test(cache, X[ex], label, grad['parameters'][0], cache['C1']['filters'], 'C1', 'filters', epsilon=1e-7) # Check gradient

			print (cost, 'Cost for a single example')
			print ('Predicted label:', cache_copy['softmax'])
			print ('Actual label:', label)

			cost_history.append(cost)

			momentum_params = 0.9 * weighted_avg_history[-1]['parameters'] + 0.1 * grad['parameters']
			momentum_bias = 0.9 * weighted_avg_history[-1]['biases'] + 0.1 * grad['biases']

			cache = update_weights(momentum_params, momentum_bias, cache_copy, lr)

			weighted_avg_history.append({'parameters': momentum_params, 'biases': momentum_bias})

		# lib.visualizeCost(cost_history)	

		lr = lib.lr_decay(lr, 1, epoch + 1) # Update learning rate
	
	lib.visualizeCost(cost_history)
	predict(cache, x_test, y_test)

	

sgd_online(x_train, y_train, 5)



	








