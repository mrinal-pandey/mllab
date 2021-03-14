# Write a program to implement k-Nearest Neighbour algorithm to classify the iris data set. Print both correct and wrong predictions. Java/Python ML library classes can be used for this problem.

from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5)
knn = KNeighborsClassifier(n_neighbors = 4)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

print("Prediction:")
print("Sample  Predicted  Expected  VERDICT")
for i in range(len(y_pred)):
    if y_pred[i] == y_test[i]:
        print(i + 1 , "\t",  y_test[i], "\t", y_pred[i], "\t", 'Correct')
    else:
        print(i + 1, "\t", y_test[i], "\t", y_pred[i], "\t", 'Wrong')

print()
print("Classification report:")
print(classification_report(y_test, y_pred))
print()
print("Accuracy score:")
print(accuracy_score(y_test, y_pred))
print()
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))
