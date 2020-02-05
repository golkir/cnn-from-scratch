# import numpy as np 

# def train_batch_small (training_set, labels, epochs):
# 	lr = 0.2
# 	weighted_avg_history = [{'parameters':0, 'biases': 0}]
# 	cost_history = []

# 	cache = { 'C1': {'fmaps': None, 'bias': init_weights((8,1,1), 784, 6272), 'filters': init_weights((8,3,3), 784, 6272), 'maxpool': None, 'sigmoid': None },
# 	          'C2': {'fmaps': None, 'filters': init_weights((16,5,5), 1568, 1600), 'bias': init_weights((16,1,1), 1568, 1600), 'maxpool': None, 'sigmoid': None },
# 	          'C3': {'fmaps': None, 'filters': init_weights((120,400), 400, 120), 'bias': init_weights((120,1), 400, 120), 'sigmoid': None},
# 	          'F6': {'fmaps': None, 'filters': init_weights((64,120), 120, 64), 'bias': init_weights((64,1), 64,120), 'sigmoid': None},
# 	          'F7': {'fmaps': None, 'filters': init_weights((10,64),64,10), 'bias': init_weights((10,1),64,10)},
# 	          'softmax': None
# 	}

# 	N = training_set.shape[0]
# 	for epoch in range(epochs):
# 		permutation = np.random.permutation(N)
# 		X, Y = training_set[permutation], labels[permutation]
# 		print('Epoch:', epoch)
# 		batch_grads_params = []
# 		batch_grads_biases = []
# 		epoch_cost = []
# 		for image in range(len(X)):
# 			X[image] = lib.image_normalize(X[image])
# 		for example in range(len(X)):
# 			# print (Y[example], 'Label of example')
# 			# im = np.array(X[example], dtype='float')
# 			# plt.imshow(im, cmap='gray')
# 			# plt.show()
# 			cache_copy = copy.deepcopy(cache)
# 			cache_copy = forwardprop(X[example], cache_copy)
# 			# if cache['C1']['fmaps'] is not None and cache_copy['C1']['fmaps'] is not None:
# 			# 	print (cache['C1']['fmaps'] - cache_copy['C1']['fmaps'], 'Difference betweten current and previous C1 fmaps')
# 			loss_gr = cache_copy['softmax'] - Y[example]
# 			print ( 'Current learning rate', lr)
# 			print('Epoch', epoch)
# 			print(np.mean(cost_history), 'Mean of entire cost history')
# 			print ('Predicted label:', cache_copy['softmax'])
# 			print ('Actual label:', Y[example])
# 			ex_loss = log_loss(Y[example].reshape(1,10), cache_copy['softmax'])
# 			print(ex_loss, 'Loss for example')
# 			epoch_cost.append(ex_loss)
# 			example_grad = backprop(cache_copy, X[example], Y[example], loss_gr)
# 			if batch_grads_params:
# 				batch_grads_params[0] += example_grad['parameters']
# 			else:
# 				batch_grads_params.append(example_grad['parameters'])
# 			if batch_grads_biases:
# 				batch_grads_biases[0] += example_grad['biases']
# 			else:
# 				batch_grads_biases.append(example_grad['biases'])
# 		print('Avg epoch cost:', np.mean(epoch_cost))
# 		cost_history.append(np.mean(epoch_cost))
# 		momentum_params = 0.9 * weighted_avg_history[-1]['parameters'] + 0.1 * batch_grads_params[0]
# 		momentum_bias = 0.9 * weighted_avg_history[-1]['biases'] + 0.1 * batch_grads_biases[0]
# 		cache = update_weights(momentum_params, momentum_bias, cache_copy, lr)
# 		weighted_avg_history.append({'parameters': momentum_params, 'biases': momentum_bias})

# 	visualizeCost(cost_history)
# 	predict(cache, test_set, test_labels )
	

# def train (training_set, labels, batch_size, epochs):
# 	lr = 0.1
# 	weighted_avg_history = [{'parameters':0, 'biases': 0}]
# 	cost_history = []

# 	cache = { 'C1': {'fmaps': None, 'bias': np.zeros((8,1,1)), 'filters': init_weights((8,3,3), 784, 6272), 'maxpool': None, 'sigmoid': None },
# 	          'C2': {'fmaps': None, 'filters': init_weights((16,5,5), 1568, 1600), 'bias': np.zeros((16,1,1)), 'maxpool': None, 'sigmoid': None },
# 	          'C3': {'fmaps': None, 'filters': init_weights((120,400), 400, 120), 'bias': np.zeros((120,1)), 'sigmoid': None},
# 	          'F6': {'fmaps': None, 'filters': init_weights((84,120), 120, 84), 'bias': np.zeros((84,1)), 'sigmoid': None},
# 	          'F7': {'fmaps': None, 'filters': init_weights((10,84),84,10), 'bias': np.zeros((10,1))},
# 	          'softmax': None
# 	}

# 	N = training_set.shape[0]

# 	for epoch in range(epochs):
# 		permutation = np.random.permutation(N)
# 		X, Y = training_set[permutation], labels[permutation]
# 		minibatches_X = [ X[k : k + batch_size] for k in range(0, N, batch_size)]
# 		minibatches_Y = [ Y[k : k + batch_size] for k in range(0, N, batch_size)]

# 		for minibatch in range(len(minibatches_X)):
# 			print('Epoch:', epoch, 'Minibatch:', minibatch)
# 			if (minibatch in [400]):
# 				visualizeCost(cost_history)
# 			fprops = []
# 			loss_grads = []
# 			minibatch_loss = 0
# 			minibatch_grads_params = []
# 			minibatch_grads_biases = []
# 			for example in range(len(minibatches_X[minibatch])):
# 				fprop = forwardprop(lib.image_normalize(minibatches_X[minibatch][example]), copy.deepcopy(cache))
# 				if loss_grads:
# 					loss_grads[0] += fprop['softmax'] - minibatches_Y[minibatch][example]
# 				else: 
# 					loss_grads.append(fprop['softmax'] - minibatches_Y[minibatch][example])
# 				fprops.append(fprop)
# 				print ('Predicted label:', fprop['softmax'])
# 				print ('Actual label:', minibatches_Y[minibatch][example])
			
# 			loss_grads[0] = loss_grads[0] / batch_size

# 			for example in range(len(fprops)):
# 				ex_loss = log_loss(minibatches_Y[minibatch][example].reshape(1,10), fprops[example]['softmax'])
# 				minibatch_loss += ex_loss
# 			avg_minibatch_loss = minibatch_loss / len(minibatches_X[minibatch])

# 			cost_history.append(avg_minibatch_loss)

# 			print(avg_minibatch_loss, 'Avergage cost for a minibatch')

# 			for example in range(len(fprops)):
# 				example_grad = backprop(fprops[example], lib.image_normalize(minibatches_X[minibatch][example]), minibatches_Y[minibatch][example], loss_grads[0])
# 				if minibatch_grads_params:
# 					minibatch_grads_params[0] += example_grad['parameters']
# 				else:
# 					minibatch_grads_params.append(example_grad['parameters'])
# 				if minibatch_grads_biases:
# 					minibatch_grads_biases[0] += example_grad['biases']
# 				else:
# 					minibatch_grads_biases.append(example_grad['biases'])

# 			momentum_params = 0.9 * weighted_avg_history[-1]['parameters'] + 0.1 * minibatch_grads_params[0]
# 			momentum_bias = 0.9 * weighted_avg_history[-1]['biases'] + 0.1 * minibatch_grads_biases[0]
# 			cache = update_weights(momentum_params, momentum_bias, cache, lr)
# 			weighted_avg_history.append({'parameters': momentum_params, 'biases': momentum_bias})
			

# 		visualizeCost(cost_history)


# 	predict(cache, test_set, test_labels )

# 	# Return weights and biases of a trained model

# 	return cache 

# 	def train_small_sample(training_set, batch, labels, epochs):
# 	lr = 0.1
# 	cost_history = []
# 	cache = { 'C1': {'fmaps': None, 'bias': init_weights((8,1,1), 784, 6272), 'filters': init_weights((8,3,3), 784, 6272), 'maxpool': None, 'sigmoid': None },
# 	          'C2': {'fmaps': None, 'filters': init_weights((16,5,5), 1568, 1600), 'bias': init_weights((16,1,1), 1568, 1600), 'maxpool': None, 'sigmoid': None },
# 	          'C3': {'fmaps': None, 'filters': init_weights((120,400), 400, 120), 'bias': init_weights((120,1), 400, 120), 'sigmoid': None},
# 	          'F6': {'fmaps': None, 'filters': init_weights((64,120), 120, 64), 'bias': init_weights((64,1), 64,120), 'sigmoid': None},
# 	          'F7': {'fmaps': None, 'filters': init_weights((10,64),64,10), 'bias': init_weights((10,1),64,10)},
# 	          'softmax': None
# 	}
# 	random = np.random.randint(0, 40000)
# 	sample = training_set[random : random + batch]
# 	labels_batch = labels[random : random + batch]
# 	vis_schedule = [v for v in range(200, epochs, 500)]
# 	for epoch in range(epochs):
# 		if epoch in vis_schedule:
# 			visualizeCost(cost_history)	
# 		if lr > 0.01:
# 			lr = lr - lr / 500
# 		fprops = []
# 		for ex in range(len(sample)):
# 			print ('Epoch:', epoch, 'Example:', ex)
# 			print (lr, 'Current learning rate')
# 			image = image_prepare(sample[ex], zeropad = True)
# 			cache = forwardprop(image, cache)
# 			loss_gr = cache['softmax'] - labels_batch[ex]
# 			grad = backprop(cache, image, labels_batch[ex], loss_gr)
# 			cost = log_loss(labels_batch[ex].reshape(1,10), cache['softmax'])
# 			print (cost, 'Cost for a single example')
# 			print ('Predicted label:', cache['softmax'])
# 			print ('Actual label:', labels_batch[ex])
# 			cost_history.append(cost)
# 			cache = update_weights(grad['parameters'], grad['biases'], cache,lr)
# 	visualizeCost(cost_history)	
# 	predict(cache, test_set, test_labels)