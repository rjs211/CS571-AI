# Import the required libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re, random
from copy import copy

# Download nltk stopwords corpus.
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Read the file line by line and clean the text of punctuation.
with open('all_sentiment_shuffled.txt', 'r') as file:
    data = file.readlines()
    data = [re.sub(r'([^\w\s]|[0-9])', ' ', line) for line in data]
    data = [re.sub(r'(\s+)', ' ', line) for line in data]

# Split the data column-wise.
split_data = [(line2[0], line2[1], line2[2], line2[3:]) for line2 in [line1.strip().split() for line1 in data]]

# Create the text and label data, and clear the stopwords from the text.
X = [line[3] for line in split_data]
Y = [line[1] for line in split_data]

stop_words = set(stopwords.words('english'))
stop_words.add('')

X = [[w for w in words if w not in stop_words] for words in X]

# Function for generating the bag-of-words vocabulary.
def get_vocab(XData):
    vocab = set()

    for line in XData:
        for word in line:
            vocab.add(word)
            
    return vocab

# Function for generating hashmap with word counts in individual documents.
def document_set(document):
    document_with_counts = {}

    for word in document:
        if word not in document_with_counts.keys():
            document_with_counts[word] = 1
        else:
            document_with_counts[word] += 1

    return document_with_counts

# Function for calculating the prior and observed probablity values.
def TrainNaiveBayes(XTrain, YTrain, alpha = 1):
    prior = {}
    vocabulary = get_vocab(XTrain)
    prob_word_given_class = {}
    classes = set(YTrain)

    for c in classes:
        prior[c] = np.log(len([y for y in YTrain if y == c]) / len(YTrain))
        class_documents = [doc for doc, label in zip(XTrain, YTrain) if label == c]
        class_documents_with_count = [document_set(doc) for doc in class_documents]
        total_word_count = sum([len(doc) for doc in class_documents])
        prob_word_given_class[c] = {}

        for word in vocabulary:
            word_occurences = 0

            for doc in class_documents_with_count:
                if word in doc.keys():
                    word_occurences += doc[word]

            prob_word_given_class[c][word] = np.log((word_occurences + alpha) / (total_word_count + alpha * len(vocabulary)))
    
    return prior, prob_word_given_class, vocabulary

# Function for predicting the label of a given text given the
# previously generated probability values.
def PredNaiveBayes(XTest, prior, prob_word_given_class, vocabulary):
    pred_labels = []
    
    for line in XTest:
        posterior = {}
        max_line = -float('inf')
        argmax_line = None

        for c in prior.keys():
            posterior[c] = prior[c]    
            
            for word in line:
                if word in vocabulary:
                    posterior[c] += prob_word_given_class[c][word]

            if max_line < posterior[c]:
                max_line = posterior[c]
                argmax_line = c
        
        pred_labels.append(argmax_line)
    
    return pred_labels

# Function for calculating the different scores of the prediction.
def get_scores(ytrue, ypred):
    POS_CLASS, NEG_CLASS = 'pos', 'neg'
    true_positives = len([1 for a, b in zip(ytrue, ypred) if a == POS_CLASS and b == POS_CLASS])
    false_positives = len([1 for a, b in zip(ytrue, ypred) if a == NEG_CLASS and b == POS_CLASS])
    true_negatives = len([1 for a, b in zip(ytrue, ypred) if a == NEG_CLASS and b == NEG_CLASS])
    false_negatives = len([1 for a, b in zip(ytrue, ypred) if a == POS_CLASS and b == NEG_CLASS])
    
    acc = (true_positives + true_negatives) / len(ytrue)
    if (true_positives + false_positives) != 0:
        prec = (true_positives) / (true_positives + false_positives)
    else:
        prec = float('nan')
    if (true_positives + false_negatives) != 0:
        rec = (true_positives) / (true_positives + false_negatives)
    else:
        rec = float('nan')
    if (prec + rec) != 0:
        f1 = 2. * prec * rec / (prec + rec)
    else:
        f1 = float('nan')
    
    return acc, prec, rec, f1

# Function for training and testing out the entire classifier pipeline.
def TrainTestNaiveBayes(XTrain, YTrain, XTest, YTest):
    prior, prob_word_given_class, vocabulary = TrainNaiveBayes(XTrain, YTrain)
    YPred = PredNaiveBayes(XTest, prior, prob_word_given_class, vocabulary)
    return get_scores(YTest, YPred)

# Pre-processing the data into different structures.
data = list(zip(X, Y))
random.shuffle(data)
X, Y = [d[0] for d in data], [d[1] for d in data]

# Setting the variables for k-fold cross validation.
num_folds = 5
split_size = round(len(X) / num_folds + 0.5)
X_splits = []
Y_splits = []

for i in range(num_folds):
    X_splits.append(X[i * split_size: (i + 1) * split_size])
    Y_splits.append(Y[i * split_size: (i + 1) * split_size])

# Running the classifier on the full dataset and calculating the mean
# accuracy, precision, recall and F1 scores.
all_scores = [0, 0, 0, 0]

for fold in range(num_folds):
    XTrain = copy(X_splits)
    del XTrain[fold]
    XTrain = sum(XTrain, [])
    XTest = X_splits[fold]
    YTrain = copy(Y_splits)
    del YTrain[fold]
    YTrain = sum(YTrain, [])
    YTest = Y_splits[fold]
    scores = TrainTestNaiveBayes(XTrain, YTrain, XTest, YTest)

    for i, elem in enumerate(scores):
        all_scores[i] += elem / num_folds

print('Accuracy: {} Precision: {}, Recall: {}, F1 Score: {}'.format(*all_scores))
