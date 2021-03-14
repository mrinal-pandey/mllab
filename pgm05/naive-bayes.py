# Write a program to implement the naïve Bayesian classifier for a sample training data set stored as a .CSV file. Compute the accuracy of the classifier, considering few test data sets.

import csv
import math
import random

def readCsv(filename):
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        dataset = list(reader)
        for i in range(len(dataset)):
            dataset[i] = [float(x) for x in dataset[i]]
    return dataset

def splitDataset(dataset, splitRatio):
    trainLen = int(splitRatio * len(dataset))
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainLen:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]

def separateByClass(trainSet):
    separated = {}
    for i in range(len(trainSet)):
        vector = trainSet[i]
        separated[vector[-1]] = separated.get(vector[-1], []) + [vector[:-1]]
    return separated

def mean(numbers):
    return sum(numbers) / float(len(numbers))

def stdev(numbers):
    avg = mean(numbers)
    variance = sum([math.pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)

def summarize(data):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*data)]
    del summaries[-1]
    return summaries

def getSummaries(trainSet):
    separated = separateByClass(trainSet)
    summaries = {}
    for classLabel, instance in separated.iteritems():
        summaries[classLabel] = summarize(instance)
    return summaries

def calculateProbability(x, mean, stdev):
    value = math.exp(math.pow(x - mean, 2) / float(-2 * stdev * stdev))
    return (1 / math.sqrt(2 * math.pi) * stdev) * value

def calculateClassProbabilities(testSet, summaries):
    probabilities = {}
    for classLabel, instance in summaries.iteritems():
        probabilities[classLabel] = 1
        for i in range(len(instance)):
            mean, stdev = instance[i]
            probabilities[classLabel] *= calculateProbability(testSet[i], mean, stdev)
    return probabilities

def predict(testSet, summaries):
    label, value = None, -1
    probabilities = calculateClassProbabilities(testSet, summaries)
    for classLabel, probability in probabilities.iteritems():
        if label is None and probability > value:
            label = classLabel
            value = probability
    return label

def getPredictions(testSet, summaries):
    predictions = []
    for i in range(len(testSet)):
        predictions.append(predict(testSet[i], summaries))
    return predictions

def getAccuracy(testSet, predictions):
    count = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            count += 1
    return (count / float(len(testSet))) * 100.0

dataset = readCsv('naive-bayes.csv')
trainSet, testSet = splitDataset(dataset, 0.67)
summaries = getSummaries(trainSet)
predictions = getPredictions(testSet, summaries)
accuracy = getAccuracy(testSet, predictions)
print("Length of dataset: " + str(len(dataset)))
print("Length of train set: " + str(len(trainSet)))
print("Length of test set: " + str(len(testSet)))
print("Accuracy: " + str(accuracy))
