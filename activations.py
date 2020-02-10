import numpy as np 
import math as math 
from scipy import special


def sigmoid(x):
	return special.expit(x)

def sigmoid_dt(x):
	s = sigmoid(x)
	return s * (1 - s)

def softmax (predictions):
	return special.softmax(predictions)