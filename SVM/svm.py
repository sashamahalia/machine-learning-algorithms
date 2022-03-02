import sklearn
from sklearn import datasets, svm, metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()

X = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

clf = svm.SVC(kernel="linear", C=2)
clf.fit(x_train, y_train)

y_predict = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_predict)

print(acc)