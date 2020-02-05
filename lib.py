import numpy as np
from scipy import signal
import math as math
import matplotlib.pyplot as plt

def to_onehot (arr):
	n_values = np.max(arr) + 1
	onehot = np.eye(n_values)[arr]
	return onehot

def init_weights_glorot(shape, fan_in=None, fan_out=None):
	"""
	General function to create filters/kernels of a given shape

	"""
	s = np.sqrt(2. / (fan_in + fan_out))
	return np.random.normal(loc=0.0, scale=s, size=shape)

def lr_decay(init_lr, decay_rate, epoch):
	
	return (1 / (1 + (decay_rate * epoch))  * init_lr)


def image_normalize(image):
	"""
	Normalizes an image so that all pixel values are between 0 and 1
	"""

	# Convert image to float.
	image = image.astype(np.float64)
	normalized = (image - image.min()) / (image.max() - image.min() )
	return normalized 

def image_normalize_std (image):
	"""
	Normalizing image by dividing by standard deviation. The mean is then is near zero

	Std = sqrt(Sum(x - x.mean/array_length)
	final formula: x - x.mean() / std(x)
	"""

	#i = image.astype(np.float64)
	normalized = (image - image.mean()) / np.std(image)
	return normalized 

def zeropad2D (arr,pad_size=1):
	"""
	Zero padding for array of the shape Row x Col X Pixel (12,12,1)
	"""

	padded = np.zeros((arr.shape[0] + pad_size * 2, arr.shape[1] + pad_size * 2))


	padded[pad_size: arr.shape[0] + pad_size, pad_size: arr.shape[1] + pad_size] = arr

	return padded


def sum_element_wise(arrlist):
    # assumes len(arrlist) > 0
    sum = arrlist[0].copy()
    for a in arrlist[1:]:
        sum += a
    return sum

def slidingArrSplit (arr,step):
	slices = []
	for index, item in enumerate(arr):
		if (index + step) < len(arr):
			slices.append(arr[index : index + step])
		elif (index + step) >= len(arr):
			sliceToEnd = arr[index :]
			sliceRest = arr[: step - len(sliceToEnd)]
			result = np.concatenate((sliceToEnd,sliceRest))
			slices.append(result)
	return np.asarray(slices)

def splitArray():
	arr = np.random.randint(1,33, (8,8))
	kernel = (2,2)
	rows,cols = arr.shape
	k_rows, k_cols = kernel
	col_split_indices = []
	split_size = rows // k_rows - 1

	k = k_rows
	for i in range(split_size):
		col_split_indices.append(k)
		k +=2 
	col_split = np.split(arr, col_split_indices, axis=1)
	print (arr, 'Original array')
	print (col_split, 'Splitted by columns')

	full_split = []

	for b in col_split: 
		full_split.append(np.split(b,col_split_indices,axis=0))
	print (arr, 'Original array')
	print(full_split,'Full split')
	full_split = np.asarray(full_split)
	print (full_split.shape)


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


