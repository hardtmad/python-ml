import psycopg2
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split

# import models
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

print("----------------------------------------")

# Connect to RedShift
conn_str = "dbname='dev' port='5439' user='mhardt' password='mE@Tz6BET$fg4`DY?*,j3.dn' host='prired.cplwbtzkosda.us-east-1.redshift.amazonaws.com'";
conn = psycopg2.connect(conn_str);
print("Connected to Redshift " + conn_str.split(" ")[0])

# Sql query
query = "SELECT income, num_adults, dwelling_type, single_parent FROM dsm;"
data = pd.read_sql_query(query, conn)
print("Running query " + query + "...")

# dataframe attributes
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

# build individual estimators
print('building individual estimators')
clf1 = SVC(kernel='rbf', probability=True, C=0.1, gamma=0.1)
clf2 = KNeighborsClassifier(n_neighbors=8)
clf3 = RandomForestClassifier(max_features='sqrt', n_estimators=5, max_depth=5)
print(clf1)
print(clf2)
print(clf3)

# # splitting X and y into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
#
# # fit individual models
# clf1.fit(X_train, y_train)
# clf2.fit(X_train, y_train)
# clf3.fit(X_train, y_train)

# build ensemble estimator
print('building ensemble estimator')
eclf1 = VotingClassifier(estimators=[('svc', clf1), ('knn', clf2), ('rfc', clf3)], voting='soft', weights=[2,1,1])
print(eclf1)

#splitting X and y into training and testing sets using K-Fold cross-validation
print('training and testing')
skf = StratifiedKFold(n_splits=2)
print(skf)
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    eclf1.fit(X_train, y_train)
    y_pred = eclf1.predict(X_test)
    # comparing actual response values (y_test) with predicted response values (y_pred)
    print("Ensemble metrics.accuracy_score:         ", metrics.accuracy_score(y_test, y_pred))
    #print("Ensemble metrics.matthews_corrcoef:      ", metrics.matthews_corrcoef(y_test, y_pred))
    print("Ensemble metrics.precision_recall_curve: ", metrics.precision_recall_curve(y_test, y_pred))
    print("Ensemble metrics.roc_curve:              ", metrics.roc_curve(y_test, y_pred))

    target_names = ['t', 'f']
    print(metrics.classification_report(y_test, y_pred, target_names=target_names))

print("----------------------------------------")
conn.close()
