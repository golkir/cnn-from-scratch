import numpy as np
from sklearn.metrics import log_loss  
import lib as lib
from maxpool import *
from scipy import signal
import activations 

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
	Flatten: 1X400
	Weights: 400X120
	Output: 1X120
	"""

	flatten = cache['C2']['sigmoid'].flatten().reshape(1,400)

	cache['C3']['fmaps'] = flatten.dot(cache['C3']['filters']) + cache['C3']['bias']
	
	# print(cache['C3']['fmaps'].shape , 'C3 fmaps')

	"""
	Sigmoid Activation
	Input: 1X120
	Output: 1X120
	"""

	cache['C3']['sigmoid'] = activations.sigmoid(cache['C3']['fmaps'])

	# print(cache['C3']['sigmoid'].shape, 'C3 sigmoid shape')
	
	"""
	F6: Dense layer (fully connected) 
	Input: 1X120
	Weights: 120X84
	Output: 1X84

	"""

	cache['F6']['fmaps'] = cache['C3']['sigmoid'].dot(cache['F6']['filters']) + cache['F6']['bias']

	# print(cache ['F6']['fmaps'].shape, 'F6 fmaps')

	"""
	Sigmoid activation
	Input: 1X84
	Output: 1X84
	"""

	cache['F6']['sigmoid'] = activations.sigmoid(cache['F6']['fmaps'])

	"""
	F7: Dense Layer (fully connected)
	Input: 1X84
	Weights: 84X10
	Output: 1X10
	"""

	cache['F7']['fmaps'] = cache['F6']['sigmoid'].dot(cache['F7']['filters']) + cache['F7']['bias']

	"""
	Softmax Activation
	Output: 1X10

	"""
	cache['softmax'] = activations.softmax(cache['F7']['fmaps'])

	# print(cache['softmax'].shape, 'Softmax result')

	return cache














