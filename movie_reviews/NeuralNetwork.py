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
                np.random.normal(0.0, layerDims[1][0]**-0.5, layerDims[i])
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
            
            inputLayerOutput = self.InputVector.dot(self.hiddenLayers[0])
            finalOutput = self.sigmoid(inputLayerOutput.dot(self.outputLayer))
            
            # - BACKWARDS -
            
            # FIND ERROR
            finalOutputError = finalOutput - outputHelperSI(outputs[i])
            finalOutputDelta = finalOutputError * self.sigmoid_derivative(finalOutput)
            
            inputLayerError = finalOutputDelta.dot(self.outputLayer.T)
            inputLayerDelta = inputLayerError
            
            # update weights
            self.outputLayer -= inputLayerOutput.T.dot(finalOutputDelta) * learning_rate
            self.hiddenLayers[0] -= self.InputVector.T.dot(inputLayerDelta) * learning_rate
            

            if outputHelperIS(finalOutput) == outputs[i]:
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
            
            
            
        
        
            
            
            
            
            