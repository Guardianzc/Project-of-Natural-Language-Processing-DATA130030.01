import matplotlib.pyplot as plt
from sgd import load_saved_params, sgd
from softmaxreg import softmaxRegression, getSentenceFeature, accuracy, softmax_wrapper

from data_utils import *
from nltk.corpus import stopwords


def train(trainset, tokens):
    priors = [0] * 5
    conditionalP = [[0] * len(tokens) for i in range(5)] 
    wordcount = [0] * 5
    tokencount = len(tokens)
    for ts in trainset:
        sentence = ts[0]
        sentiment = ts[1] - 1
        priors[sentiment] += 1
        wordcount[sentiment] += len(sentence)
        for word in sentence:

            conditionalP[sentiment][tokens[word]] += 1
    
    for i in range(5):
        for j in range(tokencount):
            conditionalP[i][j] = (conditionalP[i][j] + 1) / (wordcount[i] + tokencount)
    
    sumpriors = sum(priors)
    for i in range(5):
        priors[i] = priors[i] / sumpriors
        
    npconditionalP = np.array(conditionalP)
    np.savetxt('..\\datasets\\conditionalP.txt', npconditionalP)
    npprior = np.array(priors)
    np.savetxt('..\\datasets\\priors.txt', npprior)
    
    return conditionalP, priors

def test(conditionalP, priors, testset):
    
    acc = 0
    for ts in testset:
        sentence = ts[0]
        true = ts[1]
        prob = priors[::]
        for word in sentence:
            
            for i in range(5):
                prob[i] *= conditionalP[i][tokens[word]]
        
        predict = prob.index(max(prob))
        if (predict + 1) == true:
            acc += 1
    print('The accuracy is', acc/len(testset))
    return 0 

if __name__ == "__main__":
    # Load the dataset
    dataset = StanfordSentiment()
    tokens = dataset.tokens()
    nWords = len(tokens)

    # Load the train set
    trainset = dataset.getTrainSentences()
    # Prepare dev set features
    devset = dataset.getDevSentences()
       
    # Prepare test set features
    testset = dataset.getTestSentences()
    mode = "train"
    
    if mode == "train":
        conditionalP, priors = train(trainset, tokens)
        test(conditionalP, priors, testset)
    elif mode == "test":
        conditionalP = np.loadtxt('..\\datasets\\conditionalP.txt')
        priors = np.loadtxt('..\\datasets\\priors.txt')
        conditionalP = conditionalP.tolist()
        priors = priors.tolist()
        test(conditionalP, priors, testset)