# For a given set of training data examples stored in a .CSV file, implement and demonstrate the Candidate-Elimination algorithm to output a description of the set of all hypotheses consistent with the training examples.

import csv

data = []
print('Dataset:')
with open('candidate-elimination.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        data.append(row)
        print(row)
print()

N = len(data[0]) - 1
S = ['0'] * N
G = ['?'] * N

print('Most General hypothesis is', G)
print('Most Specific hypothesis is', S)
print()

S = data[0][:-1]
gset = []

for i in range(len(data)):
    if data[i][N] == 'Yes':
        for j in range(N):
            if data[i][j] != S[j]:
                S[j] = '?'
        for j in range(N):
            for k in range(1, len(gset)):
                if gset[k][j] != '?' and S[j] != gset[k][j]:
                    del gset[k]
    if data[i][N] == 'No':
        for j in range(N):
            if S[j] != '?' and data[i][j] != S[j]:
                G[j] = S[j]
                gset.append(G)
                G = ['?'] * N
    if len(gset) == 0:
        print('For training example', i + 1, 'general hypothesis', G)
    else:
        print('For training example', i + 1, 'set of general hypothesis', gset)
    print('For training example', i + 1, 'general hypothesis', S)
