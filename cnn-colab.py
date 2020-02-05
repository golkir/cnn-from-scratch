import sys 
import numpy as np 
import math as math 
import pandas as pandas 
import lib as lib
from maxpool import *
from backprop import *
import matplotlib.pyplot as plt

from mlxtend.data import loadlocal_mnist

X, y = loadlocal_mnist(
        images_path='/content/drive/My Drive/Colab Notebooks/CNN_backprop/mnist/train-images-idx3-ubyte', 
        labels_path='/content/drive/My Drive/Colab Notebooks/CNN_backprop/mnist/train-labels-idx1-ubyte')


X = X.reshape((-1,28,28))

tr_set, test_set = np.vsplit (X, [42000])[0], np.vsplit(X,[42000])[1]


tr_labels, test_labels = y[:42000], y[42000:]

tr_labels = lib.to_onehot(tr_labels)
test_labels = lib.to_onehot(test_labels)


def createFilters(shape, N_input_units):
	"""
	General function to create filters/kernels of a given shape
	"""
	np.random.seed(19111985)
	filters = np.random.normal(0,1, shape) * math.sqrt(1 / N_input_units)
	return filters
def createBias (shape, N_input_units):
	np.random.seed(19111985)
	biases = np.random.normal(0,1, shape) * math.sqrt(1 / N_input_units)
	return biases 

def convolve(input,filter, stride = 1):
	"""
	Convolution function
	@image Image or feature map to convolve with filter
	@ filter a 2D filter matrix to use
	@ bias bias to add to each convolution unit
	"""
	
	# Calculate output dimension. Assume a rectangular 2D filter and 2D image
	input_h, input_w = input.shape
	filter_h, filter_w = filter.shape 
	output_dim = input_h - filter_h // stride + 1
	
	# Initialize convolution output 
	output = np.ones((output_dim,output_dim))
	
	# Main loop
	for i_row,row in enumerate(input):
		for i_col, col in enumerate(input[i_row]):
			if  i_row  < output_dim and i_col < output_dim:
				field = input[i_row : i_row + filter_w, i_col : i_col + filter_h]
				activation = np.sum(field * filter)
				output[i_row, i_col] = activation
	return output 
	
	# Initialize convolution output 
	output = np.ones((output_dim,output_dim,1))
	# Main loop
	for i_row,row in enumerate(input):
		for i_col, col in enumerate(input[i_row]):
			if  i_row  < output_dim and i_col < output_dim:
				field = input[i_row : i_row + filter_w, i_col : i_col + filter_h]
				activation = np.sum(field * filter)
				output[i_row, i_col] = activation
	return output 

def f_con (filters,gradient):
	filters = np.ones((5,5))
	gradient = np.ones((8,8))
	zeros = np.zeros((16,16))
	zeros[filters.shape[0] - 1 : zeros.shape[0] - (filters.shape[0]-1) , filters.shape[1] - 1 : zeros.shape[1] - (filters.shape[0] - 1)] = gradient

def addBias (input,bias):
	return input + bias
	

def convolveOverMultipleInputs (input,filter, stride = 1):
	"""
	Feature maps for the C2 convolutional layer. 
	The layer has 16 filters connected to 5:5 regions in layer S2 feature maps (we have total 6 S2 feature maps).
	The region is identical for each subset of C2 filters (see below)
	They are connected as follows:
	First 6 C2 filters are connected to identical regions in three contiguous S2 feature maps.
	Table C2_1-6 (includes 6 filters)
	C2_1 -> S2_1, S2_2, S2_3 
	C2_2 - > S2_2, S2_3, S2_4
	C2_3 -> S2_3, S2_4, S2_5 
	C2_4 -> S2_4, S2_5, S2_6
	C2_5 -> S2_5, S2_6, S2_1 
	C2_6 -> S2_6, S2_1, S2_2 

	Table C2_7-12 (includes 6 filters)

	These filters are connected to identical regions in four contiguous S2 feature maps
	C2_7 -> S2_1, S2_2, S2_3, S2_4
	C2_8 -> S2_2, S2_3, S2_4, S2_5
	C2_9 -> S2_3, S2_4, S2_5, S2_6
	C2_10 -> S2_4, S2_5, S2_6, S2_1
	C2_11 -> S2_5, S2_6, S2_1, S2_2
	C2_12 -> S2_6, S2_1, S2_2, S2_3

	Table C2_13-15 (includes 3 filter)
	These filters take inputs from some discontinous subsets of 4 S2 feature maps
	For example,

	C2_13 -> S2_1, S2_2, S2_4, S2_5 
	C2_14 -> S2_2, S2_3, S2_5, S2_6
	C2_15 -> S2_1, S2_3, S2_4, S2_6

	Finally C2_16 takes input from all 6 S2 feature maps

	According to https://stats.stackexchange.com/questions/166429/convolutional-neural-network-constructing-c3-layer-from-s2:

	We need to have individual filter (separate set of 5:5 weights for each unit in the feature map)
	So, for example, C2_1 will have three 5:5 filters each for S2_1, S2_2, S2_3 
	Also, it seems that we have only one bias for each feature map in C2 rather than each for each unit in C2 feature map
	"""

	convolutions = []
	for index,value in enumerate(input):
		convolutions.append (convolve(value,filter[index],stride))
	return lib.sum_element_wise(convolutions)

def C2_backprop(input, gradient):

	"""
	C2 has the input S1 sigmoid (6X12X12 feature maps)
	C2 filteres has the following dimensions:
	- 18X5X5, 24X5X5, 12X5X5, 6X5X5
	So, we need to somehow get gradient in this dimensions. 
	Dimensions of dL_dS2_maxpool = 16X8X8 - the same as C2 feature maps

	Formula: deconvolve . Just convolve gradient over C2 inputs (C1 sigmoid). So it's like a reverse convolution
	Output should have the same dimensions as C2 filters. the Same as C2 filters. We actually convolve 8x8 S2 maxpool gradient over 12x12 inputs 
	"""

	# Split C1 sigmoid results as we did during the forward pass
	S1_split_3 = lib.slidingArrSplit(input,3)
	S1_split_4 = lib.slidingArrSplit(input,4)
	S1_split_4_assym = np.asarray( [ [ input[0],input[1],input[3],input[4] ], 
	                               [ input[1],input[2],input[4],input[5] ],
	                               [ input[0],input[2],input[3],input[5] ] ] )

	# Convolve each of first 6 gradients of C2 maxpool over corresponding three feature maps in C1 sigmoid.
	# As a result, we need to get 18 feature maps, the same number as the first filter of C2. 
	# 18X5X5

	C2_params_grad_1 =[]
	for i in range (0,6):
		for b in range (0,3):
			convolution = convolve(S1_split_3[i,b], gradient [i])
			C2_params_grad_1.append(convolution)
	C2_params_grad_1 = np.asarray(C2_params_grad_1).reshape(6,3,5,5)

	# Conveolve each of second 6 gradients of C2 maxpool over corresponding four feature maps in C1 sigmoid 
	# Output: 24 feature maps, 24X5X5

	C2_params_grad_2 = []

	for i in range (0,6):
		for b in range(0,4):
			convolution = convolve(S1_split_4[i,b], gradient[i + 6])
			C2_params_grad_2.append(convolution)
	C2_params_grad_2 = np.asarray(C2_params_grad_2).reshape(6,4,5,5)

	# Conveolve each of 12-15 gradients of C2 maxpool over corresponding four feature maps in C1 sigmoid 
	# Output: 12 feature maps, 12X5X5

	C2_params_grad_3 = []

	for i in range (0,3):
		for b in range(0,4):
			convolution = convolve(S1_split_4_assym[i,b], gradient[i + 12])
			C2_params_grad_3.append(convolution)
	C2_params_grad_3 = np.asarray(C2_params_grad_3).reshape(3,4,5,5)

	# One with all

	C2_params_grad_4 = []

	for i in range (0,6):
		convolution = convolve(input[i], gradient[15])
		C2_params_grad_4.append(convolution)
	C2_params_grad_4 = np.asarray(C2_params_grad_4).reshape(6,5,5)
	convolutions = np.asarray([C2_params_grad_1, C2_params_grad_2, C2_params_grad_3, C2_params_grad_4])

	return convolutions 


def C2_convolve(input, C2_parameters):
	
	C2 = np.zeros((16,8,8))

	S1_split_3 = lib.slidingArrSplit(input,3)
	S1_split_4 = lib.slidingArrSplit(input,4)

	# Hard to find structure here to create a function. So by hand. 
	S1_split_4_assym = [ [ input[0],input[1],input[3],input[4] ],
                         [ input[1],input[2],input[4],input[5] ],
                         [ input[0],input[2],input[3],input[5] ] ]       

    # Convolve first 18 filters which will produce 6 feature maps. 
	for subset_index, subset in enumerate(S1_split_3):
		for filter_index in range(0,6): 
				convolutions = convolveOverMultipleInputs(subset,C2_parameters[0][filter_index])
				C2[subset_index] = convolutions

	# Next six feature maps are connected to identical regions in four contiguous S2 feature maps

	for subset_index, subset in enumerate(S1_split_4):
		for filter_index in range (0, 6):
				convolutions = convolveOverMultipleInputs(subset, C2_parameters[1][filter_index])
				C2[6 : 6 + subset_index] = convolutions

	# Next, assymetrical 
	# C2_13 -> S2_1, S2_2, S2_4, S2_5 
	# C2_14 -> S2_2, S2_3, S2_5, S2_6
	# C2_15 -> S2_1, S2_3, S2_4, S2_6


	for subset_index, subset in enumerate(S1_split_4_assym):
		for filter_index in range (0, 3):
				convolutions = convolveOverMultipleInputs(subset, C2_parameters[2][filter_index])
				C2[12 : 12 + subset_index] = convolutions 

	# Final fully connected feature map 

	C2_16_convolution = convolveOverMultipleInputs(input, C2_parameters[3])
	# Add bias
	C2[15] = C2_16_convolution
	# Add bias
	C2 += C2_parameters[-1]

	return C2 

		
# Initialize convolutional parameters


def forwardprop(image,cache):


	"""
	Stochastic Gradient Descent (SGD) on one image (study caveats)

	"""

	"""
	Image min/max normalization. Final values are between o and 1

	Formula: X - X.min / X.max - X.min (substracting min and max for the case if 0 and 255 are not the min and max value in an image)
	"""

	image = lib.image_normalize(image)


	"""
	 C1 convolution layer. Six feature maps connected to 5:5 region in the image. Each filter has 25 parameters plus a trainable bias
	 Tested: yes

	"""

	cache['C1']['fmaps'] = np.zeros((6,24,24))
	for index, filter in enumerate(cache['C1']['filters']):
		cache['C1']['fmaps'][index] = convolve(image, filter)
	cache['C1']['fmaps'] += cache['C1']['bias']


	""" 
	S1 Maxpool layer. Uses 2,2 filter. The operation is not overlapping. The result is 6 12:12 feature maps 
	Tested: yes
	Add: maxpool indices to remember elements participating in maxpool

	"""

	cache['C1']['maxpool'] = maxpool_run(cache['C1']['fmaps'],(2,2))



	"""
	Element-wise application of sigmoid for each feature map in S1 
	Test: yes
	Comments: seems right, not sure why some valudes round to 1. Check if needed

	"""

	cache['C1']['sigmoid'] = np.zeros((6,12,12))

	for index, maxpool in enumerate(cache['C1']['maxpool']):
		cache['C1']['sigmoid'][index] = lib.sigmoid(maxpool)

	"""
	C2. Second convolution layer. See description in convolveOverMultipleInputs function
	Tested: yes
	Comments: shapes (8*8*1) and number of fmaps (16) seem to be right. Not sure about correctness of convolveOverMultiple function 

	"""

	cache['C2']['fmaps'] = C2_convolve(cache['C1']['sigmoid'], cache['C2']['parameters'])



	# Layer S2. Subsampling layer with 16 feature maps. Each feature map is connected to 2:2 region in C2 feature map
	# As a result, we get 4:4 S2 feature maps 
	# Tested: yes
	# Comments: seems to be correct. However, sigmoid outputs 1s everywhere. Seems to be a problem with precision. I use float64 but the precision is 15 points

	cache['C2']['maxpool'] = maxpool_run(cache['C2']['fmaps'],(2,2))



	cache['C2']['sigmoid'] = np.zeros((16,4,4))

	for index, maxpool in enumerate(cache['C2']['maxpool']):
		cache['C2']['sigmoid'][index]= lib.sigmoid(maxpool)

	"""
	C3. Convolution layer 3. Full connection with S2. Each unit in C3 is connected to 4:4 region in all 16 S2 feature maps. The output is 1:1. 
	Total: 120 feature maps

	Here we may not use convolution because the receptive field is equal to the input feature map (S2)

	Tested: yes
	Comments: seems to be correct so far. Output: 120 fmaps of shape (1x1x1). 
	Again sigmoid outputs 1 everywhere

	"""
	cache['C3']['fmaps'] = []

	for index, fmap in enumerate(cache['C3']['filters']):
		cache['C3']['fmaps'].append(convolveOverMultipleInputs(cache['C2']['sigmoid'],fmap))
	
	cache['C3']['fmaps'] = np.asarray(cache['C3']['fmaps']).reshape(120,1)

	cache['C3']['fmaps'] += cache['C3']['bias']


	# C3 sigmoid

	cache['C3']['sigmoid'] = np.zeros((120,1,1))

	for index, c3_fmap in enumerate(cache['C3']['fmaps']):
		cache['C3']['sigmoid'][index] = lib.sigmoid(c3_fmap)

	"""
	Layer F6. 

	Preparing for the softmax layer

	F6 has 10 units with filter 120 X1 each. Is fully connected to C3. We should have a bias for each unit

	Tested: yes
	Comments: seem to be correct so far. Return a 10*1 vector with class values for softmax processing

	"""



	cache['F6']['fmaps'] = cache['C3']['sigmoid'].reshape(120,1).T.dot(cache['F6']['filters'].reshape(120,10))


	cache ['F6']['fmaps'] += cache['F6']['fmaps'] + cache['F6']['bias'].T


	"""
	Softmax layer
	Tested: yes
	Comments: seems to be correct
	"""
	cache['softmax'] = lib.softmax(cache['F6']['fmaps'])


	return cache

def calculateCost(output, label):
	return lib.crossentropy_loss(output,label)

def backprop(cache, example, labels):

	"""
	Calculate gradient of loss with respect to softmax

	dL_dSoftmax = sj - yj. 

	Basically, a vector of labels minus a vector of softmax output 

	"""

	dL_dSoftmax = cache['softmax'] - labels

	dL_dSoftmax = dL_dSoftmax.reshape(1,10)


	"""
	Gradient of loss with respect to F6 parameters including bias
	dL_dF6_params = dL_dSoftmax * F6_input
	F6 input is C3_sigmoid

	Matrix dimensions: 10x1 * 120X1X1 

	Output -> 120X10
	"""



	dL_dF6_parameters = cache['C3']['sigmoid'].reshape(120,1).dot(dL_dSoftmax).reshape(120,10,1)


	# Gradient of loss with respect to F6 biases (10 biases). It equals dL_dSoftmax
	# Output: 10X1 

	dL_dF6_biases = dL_dSoftmax.T


	"""
	Gradient of loss w.r.t F6 inputs 
	Formula: dL_dSoftmax * F6_parameters
	Matrix dimensions: 1X10 .dot 10X120  -> 1X120. That seems like the input of F6. So dot?
	"""

	dL_dF6 = dL_dSoftmax.dot(cache['F6']['filters'].reshape(10,120))


	"""
	Gradient of loss w.r.t to C3_sigmoid parameters
	"""

	# First, C3_sigmoid derivative with respect to C3_sigmoid input (C3 fmaps)
	# Matrix dimensions. C3_fmaps -> 120 X 1 
	# Output -> 120 X 1

	dF6_dS3_sigmoid = lib.sigmoid_dt(cache['C3']['fmaps'].reshape(120,1))


	"""
	Gradient of loss with respect to S3 sigmoid 
	Formula: dL_dF6 * dF6_dS3_sigmoid 
	Dimensions: 120 X 1  * 120 X 1 = 120 X 1 ( needs to be Gadamard because sigmoid is element-wise)

	"""
	dL_dS3_sigmoid = dL_dF6.T * dF6_dS3_sigmoid


	"""
	Gradient of loss with respect to C3 parameters 
	Formula: dL_dS3_sigmoid * C3_inputs (C2 Sigmoid)
	Dimensions: 120 X 1 X 1  * 16 X 4 X 4   -> 120 X 16 X 4 X 4 (Gadamard ?)
	We need to multiply dl_dS3_sigmoid by each of 16 S2 feature maps to get 120X16X4X4 gradient of C3 parameters
	Strange: it works as gadamard - see below 
	"""

	dL_dS3_sigmoid = dL_dS3_sigmoid.reshape(120,1,1)

	dL_dC3_parameters = []

	for index,filter in enumerate(cache['C2']['sigmoid']):
		# filter = filter.reshape(1,4,4)
		dL_dC3_parameters.append(filter * dL_dS3_sigmoid)
	
	dL_dC3_parameters = np.asarray(dL_dC3_parameters).reshape(120,16,4,4)


	# We have 120 biases for each unit in C3. Thus, biases are simply 120 X 1 X 1

	dL_dC3_biases = dL_dS3_sigmoid.reshape(120,1)


	"""

	Gradient of loss with respect to C3 (specifically C3 inputs (S2 sigmoid))
	Formula: dL_dS3_sigmoid * cache['C3']['C3_parameters']
	Output: 120 X 1 X 1 (S3 sigmoid) * 120 * 16 * 4 * 4 
	Comment: we need to multiply S3 sigmoid gradient by each of 120 units in C3 (total 120 * 16 * 4 * 4 parameters)

	Hard to say what dimension of output to expect (????) Think about it. May be 16X4X4 ?

	"""

	C3_filters_copy = cache['C3']['filters'].copy()
	for sigmoid_index,unit in enumerate(dL_dS3_sigmoid):
		C3_filters_copy[sigmoid_index] =  C3_filters_copy[sigmoid_index] * unit
	dL_dC3 = lib.sum_element_wise(C3_filters_copy)


	"""
	Gradient of loss w.r.t to C2 sigmoid

	C2 sigmoid: 16 X 4 X 4 X 1

	Formula: dl_dC3 * C2_sigmoid (gradient)

	Output dimension: 

	"""
	dC2_dC2_sigmoid = lib.sigmoid_dt(cache['C2']['maxpool'])

	dL_dC2_sigmoid = dL_dC3 * dC2_dC2_sigmoid


	"""
	Gradient of loss with respect to C2 subsampling (S2) layer
	Comments: according to literature, the gradient need to be propagated only through units that are output of a max pooling layer.
	Solution: for each 16 feature maps of S2 a matrix with zeros for all non-max elements and ones for all max elements provided by C2
	Output: 16@8x8

	"""

	# Compute gradient of loss w.r.t S2 maxpooling
	# First, calculate the mask of C2 feature maps with max elements as 1
	# Second, upsample sigmoid derivative (dL_dC2_sigmoid)
	# Third, multiply upsample sigmoid with the mask to obtain the gradient with respect to max elements (max pooling)

	dL_dS2_maxpool = []

	for sigmoid_i, sigmoid in enumerate(dL_dC2_sigmoid):
		dt = maxpool_backprop(cache['C2']['fmaps'][sigmoid_i], (2,2), sigmoid)
		dL_dS2_maxpool.append(dt)
	dL_dS2_maxpool = np.asarray(dL_dS2_maxpool)

	

	"""
	Gradient of loss w.r.t C2 convolution layer filters

	Comments: 

	So, we need to find the gradient w.r.t C2 parameters. 
	C2 has the input S1 sigmoid (6X12X12 feature maps)
	C2 filteres has the following dimensions:
	- 18X5X5, 24X5X5, 12X5X5, 6X5X5
	So, we need to somehow get gradient in this dimensions. 

	Dimensions of dL_dS2_maxpool = 16X8X8 - the same as C2 feature maps

	Formula: deconvolve . Just convolve gradient over C2 inputs (C1 sigmoid). So it's like a reverse convolution
	Output should have the same dimensions as C2 filters. the Same as C2 filters. We actually convolve 8x8 S2 maxpool gradient over 12x12 inputs
	"""

	dL_dC2_parameters = C2_backprop(cache['C1']['sigmoid'], dL_dS2_maxpool)


	# Gradient of loss w.r.t to C2 convolution bias
	# It should be 16X1X1
	# I think we need to sum up dL_dS2_maxpool (16X8X8 along the second axis to get 16X1X1). Need to check with other resources

	dL_dC2_biases = np.array([ np.sum(dL_dS2_maxpool[i]) for i in range(dL_dS2_maxpool.shape[0]) ]).reshape(16,1,1)



	"""
	Gradient of loss w.r.t C2 inputs
	Use "full convolution" of C2 S2 Maxpool with C2 filters 8X8 convolve over 5X5
	"""

	# First, perform full convolution over first set of parameters: 18X5X5. 
	# Convolve over each tree and sum up convolutions. Total result: 6  

	dL_dC2 = fullconvolve_c2(cache['C2']['parameters'], dL_dS2_maxpool )


	"""
	Gradient of S1 sigmoid
	"""
	dC1_dS1_sigmoid = lib.sigmoid_dt(cache['C1']['maxpool'])

	"""
	Gradient of loss w.r.t to S1 sigmoid

	Dimensions: 6X12X12 * 6X12X12 (Element-wise)
	"""

	dL_dS1_sigmoid = dL_dC2 * dC1_dS1_sigmoid


	"""

	C1 S1 subsampling 

	Derivative of C1 maxpool layer

	"""

	dL_dS1_maxpool = np.asarray([ maxpool_backprop(cache['C1']['fmaps'][sigmoid_i], (2,2), sigmoid) for sigmoid_i, sigmoid in enumerate(dL_dS1_sigmoid) ])



	"""
	Gradient of loss w.r.t to C1 filters

	Convolve gradient of subsampling layer (6X24X24) over image (each)

	"""

	dL_dC1_parameters = np.asarray([convolve(example, dL_dS1_maxpool[i]) for i in range(0,dL_dS1_maxpool.shape[0])])


	dL_dC1_biases = np.array([ np.sum(dL_dS1_maxpool[i]) for i in range(dL_dS1_maxpool.shape[0]) ]).reshape(6,1,1)

	return { 'parameters': np.array([dL_dC1_parameters, dL_dC2_parameters, dL_dC3_parameters, dL_dF6_parameters ]),
	         'biases': np.array([dL_dC1_biases, dL_dC2_biases, dL_dC3_biases,dL_dF6_biases])}

def init_C2():

	# Six C2 feature maps with three filters each and one bias 

	parameters = []
	parameters.append(createFilters((6,3,5,5), 6 * 12 * 12))
	
	# Six C2 feature maps with four filters each and one bias 

	parameters.append(createFilters((6,4,5,5), 6 * 12 * 12))
	
	# Table C2_13-15 (includes 3 feature maps). These feature maps take inputs from some discontinous subsets of 4 S2 feature maps

	parameters.append(createFilters((3,4,5,5), 6 * 12 * 12))
	
	# Finally C2_16 takes input from all 6 S2 feature maps

	parameters.append(createFilters((6,5,5), 6 * 12 * 12))

	parameters.append(createBias((16,1,1), 6 * 12 * 12))

	return np.asarray(parameters )


def update_weights (grads_params, grads_biases,cache, lr):

	cache['C1']['filters'] -= lr * grads_params[0]
	cache ['C2'] ['parameters'][:4] -= lr * grads_params [1]
	cache['C3']['filters'] -= lr * grads_params[2]
	cache['F6']['filters'] -= lr * grads_params [3]
	cache ['C1']['bias'] -= lr * grads_biases[0]
	cache['C2']['parameters'][4] -= lr * grads_biases[1]
	cache['C3']['bias'] -= lr * grads_biases[2]
	cache ['F6']['bias'] -= lr * grads_biases[3]

	return cache 

def visualizeCost(history):
	x = []
	y = []
	for i,val in enumerate(history):
		x.append(i)
		y.append(val) 
	plt.plot(x,y)
	plt.xlabel('Iterations')
	plt.ylabel('Lost')
	plt.title("CNN implementation")
	plt.legend()
	plt.show()

def predict(model,test_set, test_labels):
	accuracy = 0
	for index in range(len(test_set)):
		image = lib.image_normalize(test_set[index])
		fprop = forwardprop(image, cache)
		if (np.argmax (cache['softmax'])  == np.argmax(test_labels[index])):
			print('the prediction is accurate')
			print ('the predicted value is:', cache['softmax'])
			print ('the actual value is:', test_labels[index])
			accuracy += 1 
		else:
			print('the prediction is wrong')
			print ('the predicted value is:', cache['softmax'])
			print ('the actual value is:', test_labels[index])
	model_accuracy = accuracy / len (test_set) * 100 

	print ('Model accuracy is:', model_accuracy)


def train (training_set, labels, batch_size, epochs):
	lr = 0.1
	cost_history = []

	cache = { 'C1': {'fmaps': None, 'bias':createBias((6,1,1), 6 * 28 * 28), 'filters': createFilters((6,5,5), 6 * 28 * 28), 'maxpool': None, 'sigmoid': None },
	          'C2': {'fmaps': None, 'parameters': init_C2(), 'maxpool': None, 'sigmoid': None },
	          'C3': {'fmaps': None, 'filters': createFilters((120, 16, 4, 4), 16 * 4 * 4), 'bias': createBias((120,1), 16 * 4 * 4), 'sigmoid': None},
	          'F6': {'fmaps': None, 'filters': createFilters((120,10,1), 120), 'bias': createBias((10,1),120)},
	          'softmax': None,
	          'cost': 5.0
	}

	N = training_set.shape[0]

	for epoch in range(epochs):
		permutation = np.random.permutation(N)
		X, Y = training_set[permutation], labels[permutation]
		minibatches_X = [ X[k : k + batch_size] for k in range(0, N, batch_size)]
		minibatches_Y = [ Y[k : k + batch_size] for k in range(0, N, batch_size)]
		for minibatch in range(len(minibatches_X)):
			print(minibatch,'Minibatch number')
			minibatch_cost = []
			minibatch_grads_params = []
			minibatch_grads_biases = []
			for example in range(batch_size):
				fprop = forwardprop(minibatches_X[minibatch][example], cache)
				example_grad = backprop(cache, minibatches_X[minibatch][example], minibatches_Y[minibatch][example]  )
				cost = lib.crossentropy_loss(fprop['softmax'], minibatches_Y[minibatch][example])
				minibatch_cost.append(cost)
				if minibatch_grads_params:
					minibatch_grads_params[0] += example_grad['parameters']
				else:
					minibatch_grads_params.append(example_grad['parameters'])
				if minibatch_grads_biases:
					minibatch_grads_biases[0] += example_grad ['biases']
				else:
					minibatch_grads_biases.append(example_grad['biases'])

			# Compute average grad for all examples in minibatch
			minibatch_cost =  np.mean(np.asarray(minibatch_cost))
			cost_history.append (minibatch_cost)
			print ('Average training cost is for a minibatch is:', cost)
			avg_grad_params = minibatch_grads_params[0] / batch_size 
			avg_grad_biases = minibatch_grads_biases[0] / batch_size

			cache = update_weights(avg_grad_params, avg_grad_biases, cache,lr)

			if cost_history[-1] < 0.2:
				break 

	predict(cache, test_set, test_labels )

	#visualizeCost(cost_history)

	# Return weights and biases of a trained model

	return cache 

train(tr_set,tr_labels,64,2)










	








