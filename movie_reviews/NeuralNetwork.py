import numpy as np
import time
import sys

class NeuralNetworkNLP:
    
    
    
    def __init__(self, layerDims, wordMapping):
        
        self.mapping = wordMapping
        
        np.random.seed(1)
        
        # Array content = [Row, Col]
        self.layerDims = layerDims
        
        # first and last are always input and output
        self.InputLayer = np.zeros(self.layerDims[0])
        self.outputLayer = np.zeros(self.layerDims[-1])
        
        self.hiddenLayers = []
        
        # hidden layers builder - random weights
        for i in range(1, len(self.layerDims)-1):
            self.hiddenLayers.append(
                np.random.normal(0.0, 1.0, layerDims[i])
            )
    
    def load_input_layer(self, input):
        
        # clear it
        self.InputLayer *= 0
        
        # load it
        for element in input.split(" "):
            self.InputLayer[0][self.mapping[element]] += 1
            
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
            
    def run(self, inputVector):
        
        # load
        self.load_input_layer(inputVector)
        
        # feed-forward
        lastOutput = self.InputLayer
        
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
    