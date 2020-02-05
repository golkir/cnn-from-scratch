# import numpy as np
# import cnn_lenet as cnn

# def gradient_check_test(cache, X, Y, gradients, parameters, layer_name, parameter_type, epsilon=1e-7):
# 	grad_approx = []
# 	for i in range(len(parameters.flatten())):
# 		thetaplus = parameters.copy().flatten()
# 		thetaplus[i] = thetaplus[i] + epsilon
# 		cache_copy = cache.copy()
# 		cache_copy[layer_name][parameter_type] = thetaplus.reshape(parameters.shape)
# 		cache_copy = forwardprop(X, cache_copy)
# 		loss_thetaplus = log_loss(Y.reshape(1,10), cache_copy['softmax'])
# 		thetaminus = parameters.copy().flatten()
# 		thetaminus[i] = thetaminus[i] - epsilon
# 		cache_copy_2 = cache.copy()
# 		cache_copy_2[layer_name][parameter_type] = thetaminus.reshape(parameters.shape)
# 		cache_copy_2 = forwardprop(X, cache_copy_2)
# 		loss_thetaminus = log_loss(Y.reshape(1,10), cache_copy_2['softmax'])
# 		dt = (loss_thetaplus  - loss_thetaminus) / (epsilon * 2)
# 		grad_approx.append(dt)
# 	grad_approx = np.array(grad_approx).reshape(parameters.shape)
# 	# print(grad_approx,'Approximated grad')
# 	# print (gradients, 'Computed gradient')
# 	numerator = np.linalg.norm(gradients - grad_approx)                                    
# 	denominator = np.linalg.norm(gradients) + np.linalg.norm(grad_approx)                
# 	difference = numerator / denominator  
# 	if difference > 1e-7:
# 		print("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
# 	else:
# 		print("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")
# 	return difference

# def forward_test():
# 	cache = { 'C1': {'fmaps': None, 'bias': np.zeros((8,1,1)), 'filters': init_weights((8,3,3), 784, 6272), 'maxpool': None, 'sigmoid': None },
# 	          'C2': {'fmaps': None, 'filters': init_weights((16,5,5), 1568, 1600), 'bias': np.zeros((16,1,1)), 'maxpool': None, 'sigmoid': None },
# 	          'C3': {'fmaps': None, 'filters': init_weights((120,400), 400, 120), 'bias': np.zeros((120,1)), 'sigmoid': None},
# 	          'F6': {'fmaps': None, 'filters': init_weights((84,120), 120, 84), 'bias': np.zeros((84,1)), 'sigmoid': None},
# 	          'F7': {'fmaps': None, 'filters': init_weights((10,84),84,10), 'bias': np.zeros((10,1))},
# 	          'softmax': None
# 	}
# 	shuffle = np.random.permutation(5)
# 	images = tr_set[shuffle]
# 	labels = tr_labels[shuffle]
# 	# for image in range(len(images)):
# 	# 	images[image] = lib.image_normalize(images[image])
	
# 	for image in range(len(images)):
# 		print (forwardprop(images[image], cache)['softmax'], 'Predicted value for image:', image)
# 		print(labels[image], 'Label for this image')


