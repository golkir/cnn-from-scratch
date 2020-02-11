import numpy as np
from maxpool import *
from scipy import signal
import activations 

def backprop(cache, example, label):


	"""
	Gradient of loss w.r.t to softmax

	"""

	dL_dSoftmax = cache['softmax'] - label.reshape(1,10)

	
	# print (dL_dSoftmax, 'Shape of gradient of loss w.r.t to softmax')

	"""
	Gradient of loss w.r.t to F7 parameters
	Dimensions:  84X10
	"""

	dL_dF7_parameters = cache['F6']['sigmoid'].T.dot(dL_dSoftmax)



	"""
	F7 Biases Gradient
	Output: 1X10
	"""
	dL_dF7_biases = dL_dSoftmax


	"""
	Gradient of loss with respect to F7 layer inputs
	Output: 1X84
	"""

	dL_dF7 = dL_dSoftmax.dot(cache['F7']['filters'].T)



	"""
	Local derrivative of F6 sigmoid
	Output: 1X84
	"""

	dF6_sigmoid = activations.sigmoid_dt(cache['F6']['fmaps'])


	"""
	Gradient of loss w.r.t to F6 sigmoid
	Output: 1X84 * 1X84 = 1X84 (Gadamard product)

	"""

	dL_dF6_sigmoid = dL_dF7 * dF6_sigmoid



	"""
	Gradient of loss w.r.t F6 parameters
	Output: 120X84
	"""

	dL_dF6_parameters = cache['C3']['sigmoid'].T.dot(dL_dF6_sigmoid)



	# print (dL_dF6_parameters.shape, 'Gradient of loss with respect to F7 dense layer parameters')

	"""
	Gradient of F6 biases
	"""

	dL_dF6_biases = dL_dF6_sigmoid



	# print(dL_dF6_biases.shape, 'Gradient of loss w.r.t F7 biases')

	"""
	Gradient of loss w.r.t F6 inputs 
	Output: 1X120
	"""

	dL_dF6 = dL_dF6_sigmoid.dot(cache['F6']['filters'].T)



	# print (dL_dF6.shape, 'Grad of loss w.r.t to F7 inputs')


	"""
	C3 Sigmoid Local Derrivative
	Output: 1X120
	"""

	dC3_sigmoid = activations.sigmoid_dt(cache['C3']['fmaps'])



	# print (dC3_sigmoid.shape, 'C3 Sigmoid derrivative')

	"""
	Gradient of loss with respect to C3 sigmoid 
	Output: 1X120  * 1X120 = 1X120 ( Gadamard product)

	"""
	dL_dC3_sigmoid = dL_dF6 * dC3_sigmoid



	# print(dL_dC3_sigmoid.shape, 'Gradient of loss w.r.t C3 sigmoid')

	"""
	Gradient of loss with respect to C3 parameters 
	Output: 400X120
	"""
	dL_dC3_parameters =  cache['C2']['sigmoid'].flatten().reshape(400,1).dot(dL_dC3_sigmoid)



	# print (dL_dC3_parameters.shape, 'C3 parameters gradient' )


	"""
	Gradient of C3 biases
	"""

	dL_dC3_biases = dL_dC3_sigmoid



	# print(dL_dC3_biases.shape, 'C3 biases')

	
	"""
	Gradient of loss with respect to C3 Layer Inputs
	Output: 16X5X5

	"""

	dL_dC3 = dL_dC3_sigmoid.dot(cache['C3']['filters'].T).reshape(16,5,5)



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


	#print (dL_dS2_maxpool.shape, 'Gradient of loss w.r.t to S2 subsampling layer')


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

	dL_dC2_biases = np.asarray([ np.sum(dL_dS2_maxpool[i]) for i in range(dL_dS2_maxpool.shape[0]) ]).reshape(16,1,1)



	# print (dL_dC2_biases.shape, 'Gradient of loss w.r.t to C2 bias')


	"""
	Gradient of loss w.r.t C2 inputs
	Operation:  "full convolution" of C2 Maxpool with C2 filters 
	Output: sum (16X10X10 full-convolve over 16X5X5) = 8X14X14 (same gradient for all 8 feature maps from C1)

	"""
	fconvs = []
	for g in range(dL_dS2_maxpool.shape[0]):
		fconv = signal.convolve2d(dL_dS2_maxpool[g], cache['C2']['filters'][g],  mode='full')
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

	dL_dC1_maxpool = []

	for sigmoid in range(dL_dC1_sigmoid.shape[0]):
		dt = maxpool_backprop(cache['C1']['fmaps'][sigmoid], (2,2), dL_dC1_sigmoid[sigmoid])
		dL_dC1_maxpool.append(dt)
	dL_dC1_maxpool = np.asarray(dL_dC1_maxpool)



	# print (dL_dC1_maxpool.shape, 'Gradient of loss w.r.t to C1 S1 subsampling layer')


	"""
	Gradient of loss w.r.t to C1 filters
	Output: 8X28X28 over 1X32X32 (image) = 8X5X5

	"""

	dL_dC1_parameters = [] 

	for f in range(dL_dC1_maxpool.shape[0]):
		dL_dC1_parameters.append(signal.correlate(example, dL_dC1_maxpool[f], mode='valid', method='direct'))

	dL_dC1_parameters = np.asarray(dL_dC1_parameters)



	# print (dL_dC1_parameters.shape, 'Gradient of loss w.r.t to C1 filters')

	"""
	Gradient of loss w.r.t C1 bias
	"""

	dL_dC1_biases = np.array([ np.sum(dL_dC1_maxpool[i]) for i in range(dL_dC1_maxpool.shape[0]) ]).reshape(8,1,1)



	# print (dL_dC1_biases.shape, 'Gradient of loss w.r.t to C1 biases')

	parameters = np.asarray([dL_dC1_parameters, dL_dC2_parameters, dL_dC3_parameters, dL_dF6_parameters, dL_dF7_parameters])
	biases = np.asarray([dL_dC1_biases, dL_dC2_biases, dL_dC3_biases,dL_dF6_biases, dL_dF7_biases])


	return { 'parameters': parameters, 'biases': biases}



