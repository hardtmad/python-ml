from sklearn import datasets
from sklearn.svm import SVC

#load example datasets
iris = datasets.load_iris()
digits = datasets.load_digits()
x, y = iris.data, iris.target

print (x.size)
print("-----")
print (y.size)
print("-----")

import numpy as np

rng = np.random.RandomState(0)
X = rng.rand(100, 10)
Y = rng.binomial(1, 0.5, 100)
X_test = rng.rand(5, 10)

#build estimator -- support vector machine -- support vector classification
clf = SVC()
print(clf.set_params(kernel='linear').fit(X, Y))
print(clf.predict(X_test))

print("----------")

print(clf.set_params(kernel='rbf').fit(X, Y))
print(clf.predict(X_test))

print("----------")

from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer

a = [[1,2], [2,4], [4,5], [3,2], [3,1]]
b = [0,0,1,1,2]

classif = OneVsRestClassifier(estimator=SVC(random_state=0))
print(classif.fit(a,b).predict(a))
print("----------")

b = LabelBinarizer().fit_transform(b)
print(classif.fit(a,b).predict(a))
print("----------")

from sklearn.preprocessing import MultiLabelBinarizer

b = [[0,1], [0,2], [1,3], [0,2,3], [2,4]]
b = MultiLabelBinarizer().fit_transform(b)
print(classif.fit(a,b).predict(a))
print("----------")