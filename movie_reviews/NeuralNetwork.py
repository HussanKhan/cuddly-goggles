import numpy as np
import time
import sys

class NeuralNetworkNLP:
    
    
    
    def __init__(self, layerDims, wordMapping):
        
        self.mapping = wordMapping
        
        np.random.seed(1)
        
        # Array content = [Row, Col]
        self.layerDims = layerDims
        
        # first is always input vector
        self.InputVector = np.zeros(self.layerDims[0])
        self.outputLayer = np.zeros(self.layerDims[-1])
        
        self.hiddenLayers = []
        
        # hidden layers builder - random weights
        for i in range(1, len(self.layerDims)-1):
            self.hiddenLayers.append(
                np.random.normal(0.0, 1.0, layerDims[i])
            )
    
    def load_input_layer(self, input):
        
        # clear it
        self.InputVector *= 0
        
        # load it
        for element in input.split(" "):
            self.InputVector[0][self.mapping[element]] += 1
            
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self,output):
        return output * (1 - output)
            
    def run(self, inputVector):
        
        # load
        self.load_input_layer(inputVector)
        
        # feed-forward
        lastOutput = self.InputVector
        
        for i in range(len(self.hiddenLayers)):   
            lastOutput = lastOutput.dot(self.hiddenLayers[i])
        
        # prediction
        finalOutput = self.sigmoid(lastOutput.dot(self.outputLayer))
        
        return finalOutput
        
            
    def test(self, testInputs, testLabels, outputHelper):
        
        correct = 0
        
        # to track how many prediction from second
        start = time.time()
        
        for i in range(len(testInputs)):
            
            prediction = self.run(testInputs[i])
            
            if outputHelper(prediction) == testLabels[i]:
                correct += 1
                
            
            # For debug purposes
            elapsed_time = float(time.time() - start)
            per_second = i / elapsed_time if elapsed_time > 0 else 0
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(testInputs)))[:4] \
                             + "% Speed(reviews/sec):" + str(per_second)[0:5] \
                             + " #Correct:" + str(correct) + " #Tested:" + str(i+1) \
                             + " Testing Accuracy:" + str(correct * 100 / float(i+1))[:4] + "%")
    
    def train(self, inputs, outputs, outputHelperSI, outputHelperIS, learning_rate):
        
        correct = 0
        
        start = time.time()
        
        for i in range(len(inputs)):
            
            # - FORWARD PASS -
            self.load_input_layer(inputs[i])            
            
            layer_1_input_hidden = self.InputVector.dot(self.hiddenLayers[0])
            layer_2_hiddin_output = self.sigmoid(layer_1_input_hidden.dot(self.outputLayer))
            
            # FIND ERROR
            layer_2_error_hidden_output = layer_2_hiddin_output - outputHelperSI(outputs[i])
            layer_2_error_delta = layer_2_error_hidden_output * self.sigmoid_derivative(layer_2_hiddin_output)
            
            # BACK PROPAGATION
            layer_1_input_hidden_error = layer_2_error_delta.dot(self.outputLayer.T)
            layer_1_error_delta = layer_1_input_hidden_error
            
            # update weights
            self.outputLayer -= layer_1_input_hidden.T.dot(layer_2_error_delta) * learning_rate
            self.hiddenLayers[0] -= self.InputVector.T.dot(layer_1_error_delta) * learning_rate
            

            if outputHelperIS(layer_2_hiddin_output) == outputs[i]:
                correct += 1
                
            # For debug purposes, print out our prediction accuracy and speed 
            # throughout the training process. 
            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(inputs)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct) + " #Trained:" + str(i+1) \
                             + " Training Accuracy:" + str(correct * 100 / float(i+1))[:4] + "%")
            if(i % 2500 == 0):
                print("")
            
            
            
        
        
            
            
            
            
            