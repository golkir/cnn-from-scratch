import numpy as np
import math as math
from operation import *

def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)

# Graph 

class Graph:
    def __init__(self, opts):
        self.layers = []
        self.alpha = 0.01
        self.grad_table = {}
        for k,v in opts.items():
            setattr (self, k, v)
    def addLayer (self,layer):
        self.layers.append(layer)
        layer.index = len(self.layers) - 1
    
    # MSE Cost function
    def MSECost(self, output, target):
        # For 1D examples
        m = len(target)
        self.cost = 1/2 * m * np.sum((target - output )**2)
        #self.cost = np.linalg.norm(cost)
        print (self.cost, 'Training cost')
        #print(norm,'Cost norm')
        costDrvt = output - target
        return  costDrvt

        
    def train(self):
        self.cost = 100 
        while self.cost > 0.1:
            self.testForward(self.layers)
            drvt = self.MSECost(self.layers[-1].value,self.target)
            self.testB(drvt)
        if self.cost <= 0.1:
            print ('Model converged')
            for layer in self.layers:
                if hasattr(layer,'weights'):
                    print (layer.weights,'Final weights')

    def gradientDescent(self, grad_table):
        for layer,weightsGrad in grad_table['weights'].items():
            self.layers[layer].weights = self.layers[layer].weights - np.round((self.alpha * weightsGrad),decimals=1)
            print(self.layers[layer].weights)
        for layer, biasGrad in grad_table['bias'].items():
            self.layers[layer].bias -= np.round((self.alpha * biasGrad),decimals=1)
            print (self.layers[layer].bias)
        self.forwardProp(self.layers)
    
    def testB(self,cost):

        ts_size = len(self.target)
        print(ts_size,'Training set size')
        l1 = self.layers[0]
        l2 = self.layers[1]
        l3 = self.layers[2]

        print (cost, 'current cost vector')
        print (l3.value, 'Output value')
        print (l2.value, 'Relu value')
        print (l1.value, 'Input value')
        print (l1.weights, 'Input layer weights')
        print (l3.weights, 'Output layer weights')

       # By now, we assume that output weights grad is calculated as follows:
       # Relu.T.dot(cost)

        outputWeights = l3.op.bprop(l2.value,l3,cost,is_weights='true')

        print (outputWeights,'Output weights grad')


        # Compute gradient of L with respect to Relu hidden layer
        # dL/dRelu = d/dL.dot(V.T)

        dLoss_dRelu =  l2.op.bprop(l1.value, l3, cost,dLoss_dRelu='True')

        print (dLoss_dRelu, 'gradient of L with respect to Relu hidden layer' )

        # Compute gradient of Relu with respect to Input layer result
        # dRelu/dInput = dLoss_dRelu  * d/dRelu (derivative of relu) (Gadamard product)

        dRelu_dInput = l2.op.bprop(l1.value, l1, dLoss_dRelu, dRelu_dInput='True' )

        print (dRelu_dInput, 'Gradient of Relu with respect to Input ' )

        inputWeights = self.trainingSet.T.dot(dRelu_dInput) # may be the absence of bias is a problem. the question what to do if i make bias fixed

        print(inputWeights, 'Gradient of loss with respect to input weights')
        
        #biasG =  np.sum(dLoss_dRelu,axis=0)

        #print(layer1.bias, 'Gradient of L w.r.t bias of the input layer')
        print(l1.bias, 'Layer 1 bias. Dont change it for now')

        
        l3.weights = l3.weights - (self.alpha * outputWeights)
        l1.weights = l1.weights - (self.alpha * inputWeights)

        #l1.bias = l1.bias - (self.alpha * biasG)
        
    def testForward(self,layers):
        data = self.trainingSet
        for l in layers:
            l.value = l.op.f(data,l)
            data = l.value
            l.fpropHistory.append(l.value)

    # Compute values for all layers in the graph -- forward propagation
    def forwardProp (self,layers): 
        data = self.trainingSet
        for l in layers:
            l.value = l.op.f(data,l)
            data = l.value
            l.fpropHistory.append(l.value)
        self.MSECost(layers[-1].value,self.target)

    # Outer skeleton of backprop
    # T - the target set of variables whose gradients are to be computed
    # G - computational graph 
    # z - the variable to be differentiated
    # grad_table is a dictionary. The '0' key is reserved for loss. 

    def backprop(self,cost):
        outputLayer = self.layers[-1]
        outputLayerInput = outputLayer.get_inputs(self)
        # Compute gradient on the output of the last layer
        outputGrad = outputLayer.op.bprop(outputLayerInput, outputLayer, cost, is_cost='true')

        # Compute grad for parameters from hidden to output layer

        outputParametersGrad = outputLayer.op.bprop(outputLayerInput,outputLayer,cost, is_parameter='true')

        # Set variable (cost) to be differentiated to 1
        grad_table = {}

        # Grad table for weights. Think how to make the code better 

        grad_table['activations'] = {'cost':1}
        grad_table ['weights'] = {}
        grad_table['activations'][outputLayer.index] = outputGrad
        grad_table['weights'][outputLayer.index] = outputParametersGrad
        grad_table['bias'] = {}
        

        # Start iterating from the top layer
        # Excludes output layer gradient already computed
        Gprime =  self.layers[: -1]
        for layer in reversed(Gprime):
                self.build_grad(layer, grad_table['activations'])

        # Calculate gradient on weights
        for layer in Gprime:
            if hasattr(layer,'weights'):
                print(layer.index,'this layer has weights')
                inputs = layer.get_inputs(self)
                grad_activation = grad_table['activations'][layer.index]
                weights_grad = layer.op.bprop(inputs,layer,grad_activation,is_parameter='true')
                grad_table['weights'][layer.index] = weights_grad
            print(grad_table['weights'])
        
        for layer in Gprime:
            if hasattr(layer,'bias'):
                grad_activation = grad_table['activations'][layer.index]
                bias_grad = layer.op.bprop(None,layer,grad_activation,is_bias='true')
                grad_table['bias'][layer.index] = bias_grad

        print(grad_table)

        # Start gradient descent

        self.gradientDescent(grad_table)


    # Inner loop/subroutine for calculating the gradient    

    # V - the variable whose gradient should be added to G and grad_table: 
    # G -  the graph  
    # B - the restriction of G to nodes that participate in the gradient
    # grad_table -  a data structure mapping nodes to their gradients

    def build_grad (self, layer, grad_table):

    # If the gradient on layer is already computed, return the grad table
        if layer.index in grad_table:
            return grad_table[layer.index]
    # Initialize Gradient for this layer
        G = []
        i = 0

        for C in layer.get_children(self):
            print(C.index,'Layer index inside build_grad')
            op = C.op
            # Recursively build grad for each child 
            D = self.build_grad(C,grad_table)
            # Get inputs from the first parent 
            inputs = layer.get_inputs(self)

            # Find grad values of all child nodes
            G.append(op.bprop (inputs, layer, D, is_cost=0))  
        print (G,'gradient')

        G = sum(G)

        print (G, 'sum')

        grad_table[layer.index] = G
       
        print (grad_table)
        
        return G


class Layer:
    type = 'Layer'
    def __init__ (self, opts):
        for k,v in opts.items():
            setattr(self,k,v)
        self.fpropHistory = []
    def __getitem__(self,key):
        return getattr(self,key)
    def getParents(self,graph):
        self.parents = graph.layers[: self.index]
        if (len(self.parents)) == 0:
            return self
        return self.parents
    def get_inputs(self, graph):
        parents = graph.layers[: self.index]
        if len(parents) > 0:
            return parents[0].value
        else:
            return self.value  
    def get_children (self, graph):
        self.children = graph.layers[self.index+1 :]
        return self.children 
    def setOp(self,op):
        self.op = Operations[op] 



