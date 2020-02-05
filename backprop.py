
import numpy as np 
from scipy import signal 

# def f_con (filters,gradient):

# 	out_sh = filters + gradient - 1
# 	outx, outy, outz = (gradient[0], out_sh, out_sh)
	
# 	fx,fy,fz = filters.shape
    
# 	zeros = np.zeros((outx, outy, outz))

# 	zeros[fx - 1 : outx - (fx - 1) , fy - 1 : outy - (fy - 1)] = gradient

# 	print(zeros)
# 	print (convolve(zeros, filters))

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

def full_conv(gradient, filter, input_shape):
	fconvs = []
	for g in range(gradient.shape[0]):
		fconv = signal.convolve(gradient[g], filter[g], mode='full', method='direct')
		fconvs.append(fconv)
	fconvs = np.sum(np.asarray(fconvs),axis=0)

	out = [fconvs for i in range(input_shape) ]

	out = np.asarray(out)
	
	return out 

def full_convolve(filters, gradient):
	
	output_shape = filters.shape[0] + gradient.shape[1] - 1

	# 7
	filter_r = filters.shape[0] - 1
	filter_c = filters.shape[1] - 1
	gradient_r = gradient.shape[0] - 1
	gradient_c = gradient.shape[1] - 1
	result = list()
	for i in range(0,output_shape):
		if (i <= filter_r):
			row_slice = (0, i + 1)
			filter_row_slice = ( 0 , i + 1)
		elif ( i > filter_r and i <= gradient_r):
			row_slice = (i - filter_r, i + 1)
			filter_row_slice = (0, i + 1)
		else: 
			rest = abs( (output_shape - 1) -  i )
			row_slice = (gradient_r  - rest, i + 1 )
			filter_row_slice = (0 ,rest + 1)
		for b in range(0,output_shape):
			if (b <= filter_c):
				col_slice = (0, b + 1)
				filter_col_slice = (0, b+1)
			elif (b > filter_c and b <= gradient_c):
				col_slice = (b - filter_c, b + 1)
				filter_col_slice = (0, b + 1)
			else:
				rest = abs ((output_shape - 1 ) - b )
				col_slice = (gradient_r - rest , b + 1)
				filter_col_slice = (0, rest + 1)
			r = np.sum(gradient[row_slice[0] : row_slice[1], col_slice[0] : col_slice[1]] * filters[filter_row_slice[0]: filter_row_slice[1], filter_col_slice[0]: filter_col_slice[1]])
			result.append(r)
	result = np.asarray(result).reshape(output_shape,output_shape)
	return result


# def fullconvolve_c2(filters, gradient, input_maps_count = 6, affected_output_count = 10, output_shape = 12):

# 	convolution_graph = {
# 	    0: [(0,0,0,0), (0,4,2,4),  (0,5,1,5),  (1,0,0,6),   (1,3,3,9),  (1,4,2,10),  (1,5,1,11),  (2,0,0,12),   (2,2,0,14)],
# 		1: [(0,0,1,0), (0,1,0,1),  (0,5,2,5),  (1,0,1,6),  (1,1,0,7),  (1,4,3,10),  (1,5,2,11),  (2,0,1,12),   (2,1,0,13)],
# 		2: [(0,0,2,0), (0,1,1,1),  (0,2,0,2),  (1,0,2,6),  (1,1,1,7),  (1,2,0,8),  (1,5,3,11),  (2,1,1,13),   (2,2,1,14)],
# 		3: [(0,1,2,1), (0,2,1,2),  (0,3,0,3),  (1,0,3,6),  (1,1,2,7),  (1,2,1,8),  (1,3,0,9),  (2,0,2,12),   (2,2,2,14)],
# 		4: [(0,2,2,2), (0,3,1,3),  (0,4,0,4),  (1,1,3,7),  (1,2,2,8),  (1,3,1,9),  (1,4,0,10),  (2,0,3,12),   (2,1,2,13)],
# 		5: [(0,3,2,3), (0,4,1,4),  (0,5,0,5),  (1,2,3,8),  (1,3,2,9),  (1,4,1,10),  (1,5,0,11),  (2,1,3,13),   (2,2,3,14)]
# 		}
# 	full_convolutions_for_all_input_maps = []
	
# 	for input_map, connections in convolution_graph.items():
# 		full_convolutions_for_one_input_map = []
# 		for connection in connections:
# 			full_convolution = full_convolve(filters[connection[0]][connection[1]][connection[2]], gradient[connection[3]])
# 			full_convolutions_for_one_input_map.append(full_convolution)
		
# 		full_convolutions_for_all_input_maps.append(full_convolutions_for_one_input_map)

# 	for i in range(input_maps_count):
# 		full_convolutions_for_all_input_maps[i].append(full_convolve(filters[3][i], gradient[15]))

# 	input_grad = []

# 	full_convolutions_for_all_input_maps = np.asarray(full_convolutions_for_all_input_maps).reshape(input_maps_count,affected_output_count,output_shape,output_shape)


# 	for subset in full_convolutions_for_all_input_maps:
# 		summed_up_convolutions = np.zeros((12,12))
# 		for i in subset:
# 			summed_up_convolutions += i
# 		input_grad.append(summed_up_convolutions) 
# 	return np.asarray(input_grad)


def convolve_backprop_test():
	a = np.arange(150).reshape(6,5,5)
	gr = np.arange(54).reshape(6,3,3)
	fconvs1 = []
	for i in range(gr.shape[0]):
		fconv = []
		for b in range(a.shape[0]):
			convolution = signal.convolve(gr[i], a[b], mode='valid', method='direct')
			fconv.append(convolution)
		fconv = np.sum(np.asarray(fconv),axis=0)
		fconvs1.append(fconv)
	fconvs = np.asarray(fconvs1)

	print(fconvs, 'First method, direct iteration')

	# Second method 

	fconvs2 = []
	for i in range(gr.shape[0]):
		convolution = np.sum(signal.convolve(a, gr[i].reshape(1,3,3), mode='valid', method='direct'), axis=0)
		fconvs2.append(convolution)
	fconvs2 = np.asarray(fconvs2)

	print(fconvs2, 'Second method, convolving gradient over all inputs simultenously')

	print(fconvs1 - fconvs2)

def testc2():
	filters = np.random.randint(1,9, (16,5,5))
	inp = np.random.randint(1,9,(8,14,14))
	bias = np.random.randint(1,9, (16,1,1))

	# test first
	c1 = []

	for i in range(filters.shape[0]):
		fmap = np.sum(signal.convolve(inp, filters[i].reshape(1,5,5), mode='valid', method='direct'), axis=0)
		c1.append(fmap)
	c1 = np.asarray(c1) + bias

	c2 = []

	for i in range(filters.shape[0]):
		fmaps = []
		for b in range(inp.shape[0]):
			fmaps.append(signal.convolve(inp[b],filters[i], mode='valid',method='direct'))
		fmaps = np.sum(np.asarray(fmaps), axis=0)
		c2.append(fmaps)
	c2 = np.asarray(c2) + bias

	print (c1.shape, 'First method where the filter convolves over all input fmaps simultenously')
	print (c2.shape, 'Second method')
	print (c1 - c2, 'Zeros if equal')




