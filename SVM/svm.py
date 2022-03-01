import sklearn
from sklearn import datasets, svm
cancer = datasets.load_breast_cancer()

print(cancer.feature_names)
print(cancer.target_names)

X = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

print(x_train, y_train)

classes = ['malignant' 'benign']
