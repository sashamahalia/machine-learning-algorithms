import sklearn
from sklearn import datasets, svm, metrics, preprocessing
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

# cancer = datasets.load_breast_cancer()
#
# X = cancer.data
# y = cancer.target

data = pd.read_csv("../news.csv", sep=",")
data = data[["title", "text", "label"]]
predict = "label"
title_list = data['title'].values.tolist()
text_list = data['text'].values.tolist()


# title = sklearn.feature_extraction.text.CountVectorizer(data["title"])
#
vectorizer = CountVectorizer()
vectorizer2 = CountVectorizer()

title_vectorized = vectorizer.fit_transform(title_list)
text_vectorized = vectorizer2.fit_transform(text_list)

X = list(zip(title_vectorized, text_vectorized))

le = preprocessing.LabelEncoder()
label = le.fit_transform(list(data["label"]))
y = list(label)

print(vectorizer2.vocabulary_.get(u'compared'))
#
# X = list(zip(title, text))
# y = list(label)
#
#
# x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
#
# clf = svm.SVC(kernel="linear", C=2)
# clf.fit(x_train, y_train)
#
# y_predict = clf.predict(x_test)
#
# acc = metrics.accuracy_score(y_test, y_predict)
#
# print(acc)