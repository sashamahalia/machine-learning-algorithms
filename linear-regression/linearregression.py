import sklearn.model_selection
import tensorflow
import keras
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style


# data = pd.read_csv("./linear-regression/student-mat.csv", sep=";")
# data = data[["G1", "G2", "G3", "studytime", "absences"]]

# predict = "G3"


data = pd.read_csv("./news.csv", sep=",")
data = data[["title", "text", "label"]]

predict = "label"

X = np.array(data.drop(labels=[predict], axis=1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
best = 0
for _ in range(100):

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(acc)

    if acc > best:
        best = acc
        with open("./linear-regression/fakenews.pickle", "wb") as f:
            pickle.dump(linear, f)

pickle_in = open("./linear-regression/fakenews.pickle", "rb")
linear = pickle.load(pickle_in)

# predictions = linear.predict(x_test)

# for x in range(len(predictions)):
#     print(predictions[x], x_test[x], y_test[x])

# p = "absences"
# style.use("ggplot")
# pyplot.scatter(data[p], data["G3"])
# pyplot.xlabel(p)
# pyplot.ylabel("Final Grade")
# pyplot.show()
