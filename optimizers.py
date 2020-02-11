import numpy as np
import copy
from sklearn.metrics import log_loss
from forwardprop import *
from backprop  import * 
import lib as lib
import matplotlib.pyplot as plt

def gradient_check(cache, X, Y, gradients, parameters, layer_name, parameter_type, epsilon=1e-7):
	grad_approx = []
	for i in range(len(parameters.flatten())):
		thetaplus = np.copy(parameters).flatten()
		thetaplus[i] = thetaplus[i] + epsilon
		cache_copy = copy.deepcopy(cache)
		cache_copy[layer_name][parameter_type] = thetaplus.reshape(parameters.shape)
		cache_copy = forwardprop(X, cache_copy)
		loss_thetaplus = log_loss(Y.reshape(1,10), cache_copy['softmax'])
		thetaminus = np.copy(parameters).flatten()
		thetaminus[i] = thetaminus[i] - epsilon
		cache_copy_2 = copy.deepcopy(cache)
		cache_copy_2[layer_name][parameter_type] = thetaminus.reshape(parameters.shape)
		cache_copy_2 = forwardprop(X, cache_copy_2)
		loss_thetaminus = log_loss(Y.reshape(1,10), cache_copy_2['softmax'])
		dt = (loss_thetaplus  - loss_thetaminus) / (epsilon * 2 )
		grad_approx.append(dt)
	grad_approx = np.asarray(grad_approx).reshape(parameters.shape)
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


def sgd_online(training_set, labels, epochs):
	cache = { 'C1': {'fmaps': None, 'bias': np.zeros((8,1,1)), 'filters': lib.init_weights_glorot((8,3,3), 900, 6272), 'maxpool': None, 'sigmoid': None },
	          'C2': {'fmaps': None, 'filters': lib.init_weights_glorot((16,5,5), 1568, 1600), 'bias': np.zeros((16,1,1)) , 'maxpool': None, 'sigmoid': None },
	          'C3': {'fmaps': None, 'filters': lib.init_weights_glorot((400,120), 400, 120), 'bias': np.zeros((1,120)), 'sigmoid': None},
	          'F6': {'fmaps': None, 'filters': lib.init_weights_glorot((120,84), 120, 84), 'bias': np.zeros((1,84)), 'sigmoid': None},
	          'F7': {'fmaps': None, 'filters': lib.init_weights_glorot((84,10),84,10), 'bias': np.zeros((1,10))},
	          'softmax': None
	}
	N = training_set.shape[0]
	cost_history = []
	lr = 0.2
	for epoch in range(epochs):
		permutation = np.random.permutation(N) # Shuffle training set 

		X, Y = training_set[permutation], labels[permutation]
		for image in range(X.shape[0]):
			X[image] = lib.image_normalize(X[image])

		for ex in range(X.shape[0]):
			print ('Epoch:', epoch, 'Example:', ex)
			print (lr, 'Current learning rate')
			label = Y[ex]
			cache = forwardprop(X[ex], cache)
			grad = backprop(cache, X[ex], label)
			cost = log_loss(label.reshape(1,10), cache['softmax'])

			print (cost, 'Cost for a single example')
			print ('Predicted label:', cache['softmax'])
			print ('Actual label:', label)

			# gradient_check(cache, X[ex], label, grad['parameters'][4], cache['F7']['filters'], 'F7', 'filters')
			cost_history.append(cost)

			cache = update_weights(grad['parameters'], grad['biases'], cache, lr)

		lib.visualizeCost(cost_history)	

		lr = lib.lr_decay(lr, 1, epoch + 1) # Update learning rate
	
	lib.visualizeCost(cost_history)


def batch_gd (training_set, labels, epochs):
	lr = 0.5
	weighted_avg_history = [{'parameters':0, 'biases': 0}]
	cost_history = []

	cache = { 'C1': {'fmaps': None, 'bias': np.zeros((8,1,1)), 'filters': lib.init_weights_glorot((8,3,3), 900, 6272), 'maxpool': None, 'sigmoid': None },
	          'C2': {'fmaps': None, 'filters': lib.init_weights_glorot((16,5,5), 1568, 1600), 'bias': np.zeros((16,1,1)) , 'maxpool': None, 'sigmoid': None },
	          'C3': {'fmaps': None, 'filters': lib.init_weights_glorot((400,120), 400, 120), 'bias': np.zeros((1,120)), 'sigmoid': None},
	          'F6': {'fmaps': None, 'filters': lib.init_weights_glorot((120,84), 120, 84), 'bias': np.zeros((1,84)), 'sigmoid': None},
	          'F7': {'fmaps': None, 'filters': lib.init_weights_glorot((84,10),84,10), 'bias': np.zeros((1,10))},
	          'softmax': None
	}

	N = 10

	permutation = np.random.permutation(N)

	X, Y = training_set[permutation], labels[permutation]

	for epoch in range(epochs):

		batch_grads_params = []

		batch_grads_biases = []

		epoch_cost = []

		for image in range(len(X)):
			X[image] = lib.image_normalize(X[image])
		
		for example in range(len(X)):

			cache = forwardprop(X[example], cache)
			
			print('Epoch', epoch)
			# print(np.mean(cost_history), 'Mean of entire cost history')
			print ('Predicted label:', cache['softmax'])
			print ('Actual label:', Y[example])

			ex_loss = log_loss(Y[example].reshape(1,10), cache['softmax'])

			print(ex_loss, 'Loss for example')

			epoch_cost.append(ex_loss)

			example_grad = backprop(cache, X[example], Y[example])

			if batch_grads_params:
				batch_grads_params[0] += example_grad['parameters']
			else:
				batch_grads_params.append(example_grad['parameters'])
			if batch_grads_biases:
				batch_grads_biases[0] += example_grad['biases']
			else:
				batch_grads_biases.append(example_grad['biases'])

		cost_history.append(np.mean(epoch_cost))

		# momentum_params = 0.9 * weighted_avg_history[-1]['parameters'] + 0.1 * batch_grads_params[0]

		# momentum_bias = 0.9 * weighted_avg_history[-1]['biases'] + 0.1 * batch_grads_biases[0]

		cache = update_weights(batch_grads_params[0], batch_grads_biases[0], cache, lr)

		# weighted_avg_history.append({'parameters': momentum_params, 'biases': momentum_bias})

	lib.visualizeCost(cost_history)
	# predict(cache, test_set, test_labels )



