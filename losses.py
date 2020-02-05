import numpy as np

def categorical_crossentropy (output,label):
	return - (np.sum(label * np.log(output)))