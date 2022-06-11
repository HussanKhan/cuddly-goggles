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
    
    if nnOutput >= 0.50:
        return 'positive'
    else:
        return 'negative'
    
mlp = NeuralNetworkNLP( [ (1, len(vocabList)), (len(vocabList), 2), (2, 1) ], wordMap)

mlp.test(reviews[:-1000], labels[:-1000], helperFunction)
