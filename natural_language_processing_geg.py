# Natural Language Processing

# Importing usual libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)


# Cleaning the texts
import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# Process main dataset

def build_corpus(dataset):
    ps = PorterStemmer()
    corpus = []
    en_stop_words = set(stopwords.words('english'))
    for i in range(len(dataset)):
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
        review = review.replace('\\n', ' ')
        review = review.lower().split()
        review = [ps.stem(word) for word in review if word not in en_stop_words]
        review = ' '.join(review)
        corpus.append(review)
    return corpus

corpus = build_corpus(dataset)

# Exact Time results
#from time import process_time
#start = process_time()
#ps = PorterStemmer() #negligeable
#print(process_time() - start)

    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray() # creates a sparse matrix

# Evaluate sparseness with Non-0 values
# 100 - 100 * sum([X[i][j] == 0 for i in range(X.shape[0]) for j in range(X.shape[1])]) / (X.shape[0] * X.shape[1])

y = dataset.iloc[:, 1].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)




## Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train.astype(float))
#X_test = sc_X.transform(X_test.astype(float))




## Whole process for one method
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import f1_score, precision_score, recall_score
#
#method = "whatever"
#y_pred = classifier.predict(X_test)
#f1 = f1_score(y_test, y_pred)
#precision = precision_score(y_test, y_pred)
#recall = recall_score(y_test, y_pred)
#
#df_results = pd.DataFrame(columns=["Method", "Precision", "Recall", "F1 score"])
#dict_results = {"Method" : [method], "Precision" : [precision], "Recall" : [recall], "F1" : [f1]}
#df = df.append(pd.DataFrame(dict_results))



# Loop over all methods
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# Limit to specific algorithms
methods = ["logistic_regression", 
           "k-nn", 
           "naive_bayes", 
           "random_forest", 
           "svm_linear", 
           "svm_rbf", 
           "svm_sigmoid"]
# Feature scaling on that bag of words, not, or both?
scales = [False, True]

# Exemple limited to 3 methods without scaling
#methods = ["svm_linear", "svm_rbf", "svm_sigmoid"]
#scales = [False]

def ml_loop(X_train, y_train, X_test, y_test, methods=["all"], scales=[False, True]):
    if methods == ["all"] or methods == None:
        methods=["logistic_regression", 
                 "k-nn", 
                 "naive_bayes", 
                 "random_forest", 
                 "svm_linear", 
                 "svm_rbf", 
                 "svm_sigmoid"]
    
    df_results = pd.DataFrame()
    
    for scaled in scales:
        if scaled:
            # Feature Scaling
            from sklearn.preprocessing import StandardScaler
            sc_X = StandardScaler()
            X_train = sc_X.fit_transform(X_train.astype(float))
            X_test = sc_X.transform(X_test.astype(float))
    
        for method in methods:
            # Logistic Regression
            if method == "logistic_regression":
                from sklearn.linear_model import LogisticRegression
                classifier = LogisticRegression(random_state=0)
            # K-NN
            if method == "k-nn":
                from sklearn.neighbors import KNeighborsClassifier
                classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
            # Naive Bayes
            if method == "naive_bayes":
                from sklearn.naive_bayes import GaussianNB
                classifier = GaussianNB()
            # Random Forest 
            if method == "random_forest":
                from sklearn.ensemble import RandomForestClassifier
                classifier = RandomForestClassifier(n_estimators=500, criterion='entropy', random_state=0)
            
            # SVM methods
            from sklearn.svm import SVC
            # SVM linear
            if method == "svm_linear":
                classifier = SVC(kernel='linear', random_state=0)
            # SVM RBF
            if method == "svm_rbf":
                classifier = SVC(kernel='rbf', random_state=0)
            # SVM Sigmoid
            if method == "svm_sigmoid":
                classifier = SVC(kernel='sigmoid', random_state=0)
        
            # Predict with current method
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            f1 = f1_score(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            
            dict_results = {"Method" : [method], "Scaled": scaled, 
                            "Accuracy": accuracy, "Precision" : [precision], "Recall" : [recall], "F1" : [f1],
                            "CM": str(cm)}
            df_results = df_results.append(pd.DataFrame(dict_results))
            df_results = df_results.sort_values(by=('F1'), ascending=False)
    return df_results

df_results = ml_loop(X_train, y_train, X_test, y_test)
df_results = df_results[["Method","Scaled","F1", "Accuracy", "Precision", "Recall", "CM"]]

# Now have a look at the df_results dataframe to get the comparison
#TODO display a nice table with matplotlib - https://matplotlib.org/gallery/misc/table_demo.html#sphx-glr-gallery-misc-table-demo-py

