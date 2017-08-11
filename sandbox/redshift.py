import psycopg2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn import preprocessing

print("----------------------------------------")
print("---------------Starting-----------------")

#Connect to RedShift
conn_str = "dbname='dev' port='5439' user='mhardt' password='mE@Tz6BET$fg4`DY?*,j3.dn' host='prired.cplwbtzkosda.us-east-1.redshift.amazonaws.com'";
conn = psycopg2.connect(conn_str);
print("Connected to Redshift " + conn_str.split(" ")[0])

#Sql query
query = "SELECT income, num_adults, dwelling_type, single_parent FROM dsm;"
#query = "SELECT marital_status, num_children, single_parent FROM dsm LIMIT 1000;"
data = pd.read_sql_query(query, conn)
print("Running query " + query + "...")

#dataframe attributes
print(str(data.columns.size) + " columns:", (data.columns))
print("Size: " + str(data.size) + " results")
print("- - - - - - - - - - - - - - - - - - - -")

# storing the feature matrix (X) and response vector (y)
X = data[data.columns[:-1]]
y = data[data.columns[-1]]

# transform nan values
X = X.as_matrix().astype(np.float)
y = y.as_matrix().astype(np.float)
X = np.nan_to_num(X)
y = np.nan_to_num(y)

X = preprocessing.scale(X)

# splitting X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

#parameters
para1 = {'gamma':[0.1, 1, 10], 'C':[0.1, 1, 10]}
para2 = {'n_neighbors':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
para3 = {'max_depth':[5,15,25], 'n_estimators':[5,15,25], 'max_features':('sqrt','log2')}

#build estimator -- support vector machine -- support vector classification
# from sklearn.svm import SVC
# #svc = SVC(kernel='rbf', gamma=2, C=1)
# tmp1 = SVC(kernel='rbf', probability=True)
# # tune parameters
# svc = GridSearchCV(tmp1, para1, cv=3)
# # fit
# svc.fit(X_train, y_train)
# # best params
# print("Models:")
# print(svc)
# print(svc.best_params_)
# # predict
# y_pred = svc.predict(X_test)
# # comparing actual response values (y_test) with predicted response values (y_pred)
# print("RBF SVM model accuracy:  ", metrics.accuracy_score(y_test, y_pred))

#build estimator -- svm -- svc
#svc2 = SVC(kernel="linear", C=0.025)
#svc2.fit(X_train, y_train)
#y_pred = svc2.predict(X_test)
#print("Linear SVM model accuracy:                    ", metrics.accuracy_score(y_test, y_pred))

#build estimator -- knn
#from sklearn.neighbors import KNeighborsClassifier
#tmp2 = KNeighborsClassifier()
# tune parameters
#knn = GridSearchCV(tmp2, para2, cv=3)
# fit
#knn.fit(X_train, y_train)
# best params
#print("Models:")
#print(knn)
#print(knn.best_params_)
# predict
#y_pred = knn.predict(X_test)
#print("Nearest Neighbor model accuracy:              ", metrics.accuracy_score(y_test, y_pred))

#build estimator -- gpc
#from sklearn.gaussian_process import GaussianProcessClassifier
#from sklearn.gaussian_process.kernels import RBF
#gpc = GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True)
#gpc.fit(X_train, y_train)
#y_pred = gpc.predict(X_test)
#print("Gaussian Process model accuracy:             ", metrics.accuracy_score(y_test, y_pred))

#build estimator -- dtc
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(max_depth=5)
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
print("Decision Tree model accuracy:                 ", metrics.accuracy_score(y_test, y_pred))

#build estimator -- rfc
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
# tmp3 = RandomForestClassifier()
# # tune parameters
# rfc = GridSearchCV(tmp3, para3, cv=3)
# # fit
# rfc.fit(X_train, y_train)
# # best params
# print("Models:")
# print(rfc)
# print(rfc.best_params_)
# # predict
# y_pred = rfc.predict(X_test)
# print("Random Forest model accuracy:                 ", metrics.accuracy_score(y_test, y_pred))

#build estimator -- ada
#ada = AdaBoostClassifier()
#ada.fit(X_train, y_train)
#y_pred = ada.predict(X_test)
#print("Ada Boost model accuracy:                     ", metrics.accuracy_score(y_test, y_pred))

#build estimator -- mlp
#from sklearn.neural_network import MLPClassifier
#mlp = MLPClassifier(alpha=1)
#mlp.fit(X_train, y_train)
#y_pred = mlp.predict(X_test)
#print("Neural Net model accuracy:                     ", metrics.accuracy_score(y_test, y_pred))

#build estimator -- gnb
#from sklearn.naive_bayes import GaussianNB
#gnb =  GaussianNB()
#gnb.fit(X_train, y_train)
#y_pred = gnb.predict(X_test)
#print("Naive Bayes model accuracy:                   ", metrics.accuracy_score(y_test, y_pred))

#build estimator -- qda
#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#qda = QuadraticDiscriminantAnalysis()
#qda.fit(X_train, y_train)
#y_pred = qda.predict(X_test)
#print("Quadratic Discriminant Analysis model accuracy:", metrics.accuracy_score(y_test, y_pred))

print("--------------Stopping------------------")
print("----------------------------------------")
conn.close()
