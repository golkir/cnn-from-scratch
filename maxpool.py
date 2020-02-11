import numpy as np 

def iterate_regions(image):
    '''
    Generates non-overlapping 2x2 image regions to pool over.
    - image is a 2d numpy array
    '''
    h, w = image.shape
    new_h = h // 2
    new_w = w // 2
    for i in range(new_h):
    	for j in range(new_w):
    		im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
    		yield im_region, i, j

def maxpool(input,sampling_field): 
	"""
	A maxpooling function that works on 2-D image array (height,width) represented as a NumPy 3-D
	array where the lust dimension is the pixel value, and the first two are height and width

	@input - image array to maxpool 
	@filter - a filter to be applied to image array. It is a tuple of integers
	"""
	# First, let's reshape a NumPy 3D array into 2D because we just can cut the 3D dimension which shape is 1 
	rows, cols = input.shape

	# Filter 
	filter_rows, filter_cols = sampling_field
	output_1D = rows // filter_rows
	output_2D = cols // filter_cols

	output = input.reshape(output_1D, filter_rows, output_2D, filter_cols).max(axis=(1,3)).reshape(output_1D, output_2D)

	return output 

def upsample(a, upsample_size):
	return np.kron(a, np.ones(upsample_size, dtype=a.dtype))

def maxpool_backprop(inp, kernel, d_L_d_out,):
    '''
    Performs a backward pass of the maxpool layer.
    Returns the loss gradient for this layer's inputs.
    - d_L_d_out is the loss gradient for this layer's outputs.
    '''

    d_L_d_input = np.zeros(inp.shape)

    for im_region, i, j in iterate_regions(inp):
    	h, w = im_region.shape
    	amax = np.amax(im_region, axis=(0, 1))
    	for i2 in range(h):
    		for j2 in range(w):
    			if im_region[i2, j2] == amax:
    				d_L_d_input[i * 2 + i2, j * 2 + j2] = d_L_d_out[i, j]
    return d_L_d_input

def maxpool_run (inp, kernel):
	kernel_r, kernel_c = kernel 
	subsamples = np.zeros((inp.shape[0], inp.shape[1] // kernel_r, inp.shape[2] // kernel_c) )
	for index in range(inp.shape[0]):
		subsamples[index] = maxpool(inp[index],(kernel_r,kernel_c))
	return subsamples

# def maxpool_backprop(inp,sampling_size,grad):
# 	# test input 
	
# 	f_rows,f_cols = sampling_size
# 	i_rows, i_cols = inp.shape 
# 	max_mask = np.zeros(inp.shape)
# 	maxpool_dt = np.zeros(inp.shape)
# 	for i in range(0,i_rows,f_rows):
# 		for c in range(0,i_cols,f_cols):
# 			field = input[i : i + f_rows, c : c + f_cols]
# 			unravel = np.unravel_index(np.argmax(field), field.shape)
# 			max_mask [i : i + f_rows, c : c + f_cols][unravel[0],unravel[1]] = 1

# 	up = upsample(grad, sampling_size )
# 	maxpool_dt = max_mask * up 

# 	return maxpool_dt

# def maxpool_test():
# 	init = np.random.randint(1,8,(10,10))
# 	print (init, 'Input')
# 	mx = maxpool(init, (2,2))
# 	print(mx,'Maxpool')
# 	grad = np.random.randint(1,8,(5,5))
# 	print(grad,'Gradient')
# 	bp = maxpool_backprop(init, (2,2),grad)
# 	print(bp, 'Maxpool backprop')





