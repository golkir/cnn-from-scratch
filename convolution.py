import numpy as np

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
def test_convolve():
	a = np.arange(9).reshape(3,3)
	f = np.arange(4).reshape(2,2)
	c = signal.convolve2d(a,f, mode='valid')
	c1 = signal.correlate(a,f, mode='valid')
	print(a)
	print(f)
	print (c, 'convolution with valid: Scipy')
	print (c1, 'signal.correlate with valid: Scipy')
	print(convolve(a,f),'my')

def test_fc():
	gr = np.arange(16).reshape(4,4)
	f = np.arange(9).reshape(3,3)
	c = signal.convolve2d(gr,f, mode='full' )
	pad = zeropad2D(gr, pad_size = 2)
	print (pad, 'Gradient padded with zeros')
	c2 = signal.convolve(pad, f, mode='valid')
	print (gr, 'Initial gradient')
	print (f, 'Filter')
	print (c, 'Full convolution with SciPy')
	print (pad, 'Padded gradient')
	print (c2, 'Valid mode to imitate full convolution')