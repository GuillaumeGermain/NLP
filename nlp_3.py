"""
Natural Language Processing
Bag or words model

looping over vocabulary size, training size
using classical classifier algorithms



Let's check now how the algorithms perform ONLY on the new dataset
It can be considered as both a multiclass classification or a regression problem


Logistic regression options:
solver='newton-cg’
warm_start=True => continue fitting
!! Logistic regression USES regularisation (L2 with default solver)



"""


# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from time import process_time


# Import classifier methods
from sklearn.linear_model import LogisticRegression
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.naive_bayes import GaussianNB
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.svm import SVC

from nlp_util import build_corpus, print_result_head

# Importing the datasets
dataset2 = pd.read_csv('newdata/1-restaurant-train.tsv', delimiter='\t', quoting=3)

dataset2.describe()


train_size = len(dataset2)
vocab_size = 3000
dataset_work = dataset2[:train_size]

corpus = build_corpus(dataset_work['Review'])

# Creating the Bag of Words model for given train+test sets
cv = CountVectorizer(max_features=vocab_size)
X = cv.fit_transform(corpus).toarray()
y = dataset_work.iloc[:, 1].values


# Splitting the dataset into the Training set and Test set
# 2000 lines in the test set should be enough to test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 2000, random_state = 0)




# ML Loop only once here

start = process_time()
X_train_len = X_train.shape[0]

print_result_head
#print("#" * 20 + "\nSTART ML_LOOP")
#print("training set size:", X_train_len)
#print("vocabulary:", X_train.shape[1])
#print("test set size:", X_test.shape[0], "\n")
#    
## Print the table header
#print("METHODS:", " ".join(methods), "\nSCALING:", scales, "\n")
#print("METHOD".ljust(20, " "), 
#      "SCALED".ljust(10, " "),
#      "TRAIN ACC".ljust(10, " "), 
#      "TEST ACC".ljust(10, " "), 
#      "F1".ljust(10, " "), 
#      "CPU TIME (in s)")
    
df_results = pd.DataFrame() 


method = "logistic_regression"
scaled = False

##############################
start_time = process_time()
# Logistic Regression
classifier = LogisticRegression(solver='newton-cg', random_state=0)
classifier.fit(X_train, y_train)

#    # K-NN
#    if method == "k-nn":
#        classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
#        classifier.fit(X_train, y_train)
#    # Naive Bayes
#    if method == "naive_bayes":
#        classifier = GaussianNB()
#        # Naive Bayes requires a non-sparse matrix type
##                re_dense = False
##                if isinstance(X_train, csr_matrix):
##                    X_train.toarray()
##                    re_dense = True
#        classifier.fit(X_train, y_train)
##                if re_dense:
##                    X_train = sparse.csr_matrix(X_train)
#    # Random Forest 500
#    if method == "random_forest":
#        classifier = RandomForestClassifier(n_estimators=500, criterion='entropy', random_state=0)
#        classifier.fit(X_train, y_train)
#    # SVM linear
#    if method == "svm_linear":
#        classifier = SVC(kernel='linear', random_state=0)
#        classifier.fit(X_train, y_train)
#    # SVM RBF
#    if method == "svm_rbf":
#        classifier = SVC(kernel='rbf', random_state=0)
#        classifier.fit(X_train, y_train)
#    # SVM Sigmoid
#    if method == "svm_sigmoid":
#        classifier = SVC(kernel='sigmoid', random_state=0)
#        classifier.fit(X_train, y_train)
#    # SVM Polynomial
#    if method == "svm_poly":
#        classifier = SVC(kernel='poly', degree=3, random_state=0)
#        classifier.fit(X_train, y_train)
    
# Predict with current method
y_pred = classifier.predict(X_test)

y_train_pred = classifier.predict(X_train)
#print("train accuracy - score", classifier.score(X_train, y_train))

def round_4(num):
    return round(float(num), 4)

# Format nicely the result and store it
train_accuracy = round(classifier.score(X_train, y_train), 4)
test_accuracy = round(accuracy_score(y_test, y_pred), 4)

# detailed % per class
f1 = list(map(round_4, f1_score(y_test, y_pred, labels=[1,2,3,4,5], average=None)))
precision = list(map(round_4, precision_score(y_test, y_pred, labels=[1,2,3,4,5], average=None)))
recall = list(map(round_4, recall_score(y_test, y_pred, labels=[1,2,3,4,5], average=None)))

cm = confusion_matrix(y_test, y_pred)
processing_time = round(process_time() - start_time, 2)

dict_results = {"Method" : [method], 
                "Scaled": scaled, 
                "TrainAccuracy": train_accuracy,
                "TestAccuracy": test_accuracy,
                "Precision" : [precision], 
                "Recall" : [recall], 
                "F1" : [f1],
                "ConfusionMatrix" : str(cm), 
                "ProcessingTime" : processing_time}
df_results = df_results.append(pd.DataFrame(dict_results))
    
print(method.lower().ljust(20, " "), 
      str(scaled).ljust(10, " "),
      str(train_accuracy).ljust(10, " "), 
      str(test_accuracy).ljust(10, " "), 
      str(f1).ljust(10, " "), 
      processing_time)

##############################

df_results = df_results.sort_values(by=('TrainAccuracy'), ascending=False)
df_results = df_results[["Method", 
                         "Scaled", 
                         "TrainAccuracy", 
                         "TestAccuracy", 
                         "F1", 
                         "Precision",
                         "Recall", 
                         "ConfusionMatrix", 
                         "ProcessingTime"]]

if verbose:
    print("\nEND ML_LOOP\n" + "#" * 20)
    print("TOTAL CPU TIME:", round(process_time() - start, 2))


#return df_results


### Display result as heatmap

import seaborn as sns 
sns.set()

class_names = ["1", "2", "3", "4", "5"]
figsize = (5,5)
fontsize=14

df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
fig = plt.figure(figsize=figsize)
try:
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
except ValueError:
    raise ValueError("Confusion matrix values must be integers.")
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
plt.ylabel('True label')
plt.xlabel('Predicted label')



