# Assuming a set of documents that need to be classified, use the naïve Bayesian Classifier model to perform this task. Built-in Java classes/API can be used to write the program. Calculate the accuracy, precision, and recall for your data set.

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB

test = fetch_20newsgroups(data_home = '20news.pkz', subset = 'test')
train = fetch_20newsgroups(data_home = '20news.pkz', subset = 'train')

print("Length of train data: " + str(len(train.data)))
print("Length of test data: " + str(len(test.data)))
print()
print("Classes:")
print(train.target_names)

cv = CountVectorizer()
tfidf = TfidfTransformer()

x_train_cv = cv.fit_transform(train.data)
x_train_tfidf = tfidf.fit_transform(x_train_cv)

model = MultinomialNB()
model.fit(x_train_tfidf, train.target)

x_test_cv = cv.transform(test.data)
x_test_tfidf = tfidf.transform(x_test_cv)

predicted = model.predict(x_test_tfidf)

print()
print("Classification report:")
print(classification_report(test.target, predicted, target_names = test.target_names))
print()
print("Confusion matrix:")
print(confusion_matrix(test.target, predicted))
print()
print("Accuracy score:")
print(accuracy_score(test.target, predicted))
