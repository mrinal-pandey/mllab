# Implement and demonstrate the FIND-S algorithm for finding the most specific hypothesis based on a given set of training data samples. Read the training data from a .CSV file.

import csv

print('Dataset:')
with open('find-s.csv') as csvfile:
    reader = csv.reader(csvfile)
    dataset = []
    for row in reader:
        dataset.append(row)
        print(row)
print()

N = len(dataset[0]) - 1
hypothesis = dataset[0][:-1]

print('Find-S algorithm:')
for i in range(len(dataset)):
    if dataset[i][N] == 'Yes':
        for j in range(N):
            if dataset[i][j] != hypothesis[j]:
                hypothesis[j] = '?'
    print('For training example', i + 1, 'hypothesis is', hypothesis)

print()
print('Maximally specific hypothesis is:')
print(hypothesis)
