
import numpy as np
import operation as op
import model as model 

X = np.array([[0,0],[0,1],[1,0],[1,1]],dtype='float64')
Y = np.array([[0],[1],[1],[0]], dtype='float64')
input_weights = np.array([[2,2],[2,2]], dtype='float64')
output_weights = np.array([[2],[3]],dtype='float64')
bias1 = np.array([0,-1],dtype='float64')
# Testing         

graph_settings = {
    'name': 'Directed acyclic graph for XOR computation',
    'trainingSet': X,
    'target': Y
}

xorNet = model.Graph(graph_settings)

inputLayer = {
    'name': 'Linear layer',
    'weights': input_weights,
    'bias': bias1 
}

hiddenLayer = {
    'name': 'Relu'
}

outputLayer = {
	'name': 'Output layer',
	'weights': output_weights
}

inputL = model.Layer(inputLayer)
hidden = model.Layer(hiddenLayer)
output = model.Layer (outputLayer)

inputL.setOp('linear')
hidden.setOp('relu')
output.setOp('linear')

xorNet.addLayer(inputL)

xorNet.addLayer(hidden)

xorNet.addLayer(output)