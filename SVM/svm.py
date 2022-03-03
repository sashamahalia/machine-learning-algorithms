import sklearn
from sklearn import datasets, svm, metrics, preprocessing
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier


# cancer = datasets.load_breast_cancer()
#
# X = cancer.data
# y = cancer.target

data = pd.read_csv("../news.csv", sep=",")
data = data[["title", "text", "label"]]
predict = "label"
text_list = data['text'].values.tolist()


# title = sklearn.feature_extraction.text.CountVectorizer(data["title"])
#
vectorizer = CountVectorizer()

text_vectorized = vectorizer.fit_transform(text_list)

# Term frequency transformer with idf (Inverse Document Frequency) enabled
tfidf_transformer = TfidfTransformer(use_idf=True)
X_train_tfidf = tfidf_transformer.fit_transform(text_vectorized)
print("X data shape: ", X_train_tfidf.shape)

le = preprocessing.LabelEncoder()
label = le.fit_transform(list(data["label"]))
y = list(label)

print(f"Occurrences of search term: ", vectorizer.vocabulary_.get(u'compared'))


#
# X = list(zip(title, text))
# y = list(label)
#

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X_train_tfidf, y, test_size=0.2)

clf = svm.SVC(kernel="linear", C=2)
clf.fit(x_train, y_train)

y_predict = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_predict)

print(acc)