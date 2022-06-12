from collections import Counter
from NeuralNetwork import NeuralNetworkNLP

# Load Data
input_file = open("./reviews.txt", 'r')
reviews = []

for review in input_file.readlines():
    reviews.append(review[:-1])

input_file.close()



output_file = open("./labels.txt", 'r')
labels = []

for label in output_file.readlines():
    labels.append(label[:-1])

output_file.close()


# Create vocab list
wordCount = Counter()
vocabList = []
wordMap = {}

for review in reviews:
    
    for word in review.split(" "):
        wordCount[word] += 1
        
        
for word, count in wordCount.most_common():
    vocabList.append(word)
    
for i, word in enumerate(vocabList):
    wordMap[word] = i


def helperFunction(nnOutput):
    
    if nnOutput >= 0.5:
        return 'positive'
    else:
        return 'negative'
    
def labelHelper(label):
    
    if label == 'negative':
        return 0
    else:
        return 1
    
                            # input vector      # input layer       # hidden   # output layer
mlp = NeuralNetworkNLP( [ (1, len(vocabList)), (len(vocabList), 10), (10, 1) ], wordMap)

#mlp.test(reviews[:-1000], labels[:-1000], helperFunction)

mlp.train(reviews[:-1000], labels[:-1000], labelHelper, helperFunction, 0.01)

#(1, 7000) (7000, 10) (10, 1)    (1, 10)