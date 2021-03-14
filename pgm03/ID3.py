# Write a program to demonstrate the working of the decision tree based ID3 algorithm. Use an appropriate data set for building the decision tree and apply this knowledge to classify a new sample.

import numpy as np
import math
import csv

class Node:
    def __init__(self, attribute):
        self.attribute = attribute
        self.children = []
        self.answer = ""
    def __str__(self):
        return self.attribute

def read_data(filename):
    with open(filename, 'r') as csvfile:
        datareader = csv.reader(csvfile)
        metadata = next(datareader)
        traindata=[]
        for row in datareader:
            traindata.append(row)
    return (metadata, traindata)

def subtables(data, col, delete):
    dict = {}
    items = np.unique(data[:, col]) # get unique values in particular column
    count = np.zeros((items.shape[0], 1), dtype=np.int32)   #number of row = number of values 
    for x in range(items.shape[0]):
        for y in range(data.shape[0]):
            if data[y, col] == items[x]:
                count[x] += 1
    #count has the data of number of times each value is present in
    for x in range(items.shape[0]):
        dict[items[x]] = np.empty((int(count[x]), data.shape[1]), dtype="|S32")
        pos = 0
        for y in range(data.shape[0]):
            if data[y, col] == items[x]:
                dict[items[x]][pos] = data[y]
                pos += 1     
        if delete:
           dict[items[x]] = np.delete(dict[items[x]], col, 1)
    return items, dict    
        
def entropy(S):
    items = np.unique(S)
    if items.size == 1:
        return 0
    counts = np.zeros((items.shape[0], 1))
    sums = 0
    for x in range(items.shape[0]):
        counts[x] = sum(S == items[x]) / (S.size)
    for count in counts:
        sums += -1 * count * math.log(count, 2)
    return sums
    
def gain_ratio(data, col):
    items, dict = subtables(data, col, delete=False) 
    #item is the unique value and dict is the data corresponding to it
    total_size = data.shape[0]
    entropies = np.zeros((items.shape[0], 1))
    for x in range(items.shape[0]):
        ratio = dict[items[x]].shape[0]/(total_size)
        entropies[x] = ratio * entropy(dict[items[x]][:, -1])
    total_entropy = entropy(data[:, -1])
    for x in range(entropies.shape[0]):
        total_entropy -= entropies[x]
    return total_entropy

def create_node(data, metadata):
    if (np.unique(data[:, -1])).shape[0] == 1:
        node = Node("")
        node.answer = np.unique(data[:, -1])
        return node
    gains = np.zeros((data.shape[1] - 1, 1))
    #size of gains= number of attribute to calculate gain
    for col in range(data.shape[1] - 1):
        gains[col] = gain_ratio(data, col)
    split = np.argmax(gains)
    node = Node(metadata[split])    
    metadata = np.delete(metadata, split, 0)
    items, dict = subtables(data, split, delete=True)
    for x in range(items.shape[0]):
        child = create_node(dict[items[x]], metadata)
        node.children.append((items[x], child))
    return node        
    
def empty(size):
    s = ""
    for x in range(size):
        s += "   "
    return s

def print_tree(node, level):
    if node.answer != "":
        print(empty(level), node.answer)
        return
    print(empty(level), node.attribute)
    for value, n in node.children:
        print(empty(level + 1), value)
        print_tree(n, level + 2)

def classify(node, data, metadata):
    if node.answer != '':
        return node.answer
    index = metadata.index(node.attribute)
    given = data[index]
    for value, n in node.children:
        if value == given or str(value)[2:-1] == given:
            return classify(n, data, metadata)
    return None

metadata, traindata = read_data("ID3.csv")
data = np.array(traindata)
node = create_node(data, metadata)
print("TREE:")
print_tree(node, 0)
print("\n----------------CLASSIFYING SAMPLE DATA------------------")
test_data = ['sunny', 'cool', 'high', 'Strong']
result = classify(node, test_data, metadata)
print("Data =", test_data)
print("Result = " + str(result)[3:-2])
test_data = ['rainy', 'hot', 'normal', 'Weak']
result = classify(node, test_data, metadata)
print("Data =", test_data)
print("Result = " + str(result)[3:-2])
