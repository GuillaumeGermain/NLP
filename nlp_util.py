#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 11 01:10:55 2018

@author: guillaume
"""

import pandas as pd



# dataset: test dataset
# dataset2: train dataset
#def build_sets(dataset_train, dataset_test, train_size, test_size, vocab_size):
#    dataset_work = dataset.append(dataset2[:train_size], ignore_index=True)
#    
#    corpus = build_corpus(dataset_work['Review'][test_size + train_size])
#    
#    # Creating the Bag of Words model
#    cv = CountVectorizer(max_features=vocab_size)
#    X = cv.fit_transform(corpus).toarray()
#    y = dataset_work.iloc[:, 1].values
#    
#    # set the matrix type to "sparse"
#    #X = sparse.csr_matrix(X)
#    
#    # Get back train and test sets form merged matrix
#    X_test = X[:test_size]
#    X_train = X[test_size:test_size + train_size]
#    y_test = y[:test_size]
#    y_train = y[test_size:test_size + train_size]
#    
#    return X_train, X_test, y_test, y_train




# Build a corpus from a text series
def build_corpus(text_series):
    import re, nltk
    try:
        from nltk.corpus import stopwords
    except:
        # Download stopwords as needed
        print("Downloading stopwords...")
        nltk.download('stopwords')
    from nltk.stem.porter import PorterStemmer
    
    # Cleaning and stemming the texts
    corpus = []
    ps = PorterStemmer()
    en_stop_words = set(stopwords.words('english'))
    for i in range(len(text_series)):
        review = re.sub('[^a-zA-Z]', ' ', text_series[i])
        review = review.replace('\\n', ' ')
        review = review.lower().split()
        review = [ps.stem(word) for word in review if word not in en_stop_words]
        review = ' '.join(review)
        corpus.append(review)
    return corpus


def ml_loop(X_train, y_train, X_test, y_test, methods=["all"], scales=[False, True], verbose=True):
    """
    Parameters:
    X_train, y_train, X_test, y_test: training and test sets
    methods: algorithms used. by default "all", all those defined in this function
    scales: feature scaling on the matrix or not. the False value (non-scaled) should be first
    verbose: display intermediate results for longer datasets/vocabularies, which may take 30 minutes to run
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
    from time import process_time
    from scipy import sparse
    from scipy.sparse.csr import csr_matrix
    #'scipy.sparse.csr.csr_matrix'

    start = process_time()
    X_train_len = X_train.shape[0]
    
    if verbose:
        print("#" * 20 + "\nSTART ML_LOOP")
        print("training set size:", X_train_len)
        print("test set size:", X_test.shape[0])
        print("vocabulary:", X_train.shape[1], "\n")
    
    if methods == ["all"] or methods is None:
        methods=["logistic_regression",
                 "k-nn",
                 "naive_bayes",
                 "random_forest",
                 "svm_linear",
                 "svm_rbf",
                 "svm_sigmoid"]
    
    if verbose:
        print("METHODS:", " ".join(methods), "\nSCALING:", scales, "\n")
        print("METHOD".ljust(20, " "), "SCALED".ljust(10, " "), "ACCURACY".ljust(10, " "), "F1".ljust(10, " "), "PROCESSING_TIME")

    df_results = pd.DataFrame() 
    
    for scaled in scales:
        if scaled:
            # The matrices have to be densified before scaling
            re_dense = False
            if isinstance(X_train, csr_matrix):
                X_train = X_train.toarray() #todense()
                X_test = X_test.toarray() #todense()
                re_dense = True
                
            # Feature Scaling
            sc_X = StandardScaler()
            X_train = sc_X.fit_transform(X_train.astype(float))
            X_test = sc_X.transform(X_test.astype(float))
                
            # Sparsify for processing
            if re_dense:
                X_train = sparse.csr_matrix(X_train)
                X_test = sparse.csr_matrix(X_test)
        
        for method in methods:
            start_time = process_time()
            # Logistic Regression
            if method == "logistic_regression":
                from sklearn.linear_model import LogisticRegression
                classifier = LogisticRegression(random_state=0)
                classifier.fit(X_train, y_train)
            # K-NN
            if method == "k-nn":
                from sklearn.neighbors import KNeighborsClassifier
                classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
                classifier.fit(X_train, y_train)
            # Naive Bayes
            if method == "naive_bayes":
                from sklearn.naive_bayes import GaussianNB
                classifier = GaussianNB()
                # Naive Bayes requires a non-sparse matrix type
                re_dense = False
                if isinstance(X_train, csr_matrix):
                    X_train.toarray()
                    re_dense = True
                classifier.fit(X_train, y_train)
                if re_dense:
                    X_train = sparse.csr_matrix(X_train)
            # Random Forest 500
            if method == "random_forest":
                from sklearn.ensemble import RandomForestClassifier
                classifier = RandomForestClassifier(n_estimators=500, criterion='entropy', random_state=0)
                classifier.fit(X_train, y_train)
            # SVM methods
            from sklearn.svm import SVC
            # SVM linear
            if method == "svm_linear":
                classifier = SVC(kernel='linear', random_state=0)
                classifier.fit(X_train, y_train)
            # SVM RBF
            if method == "svm_rbf":
                classifier = SVC(kernel='rbf', random_state=0)
                classifier.fit(X_train, y_train)
            # SVM Sigmoid
            if method == "svm_sigmoid":
                classifier = SVC(kernel='sigmoid', random_state=0)
                classifier.fit(X_train, y_train)
            
            # Predict with current method
            
            y_pred = classifier.predict(X_test)
            f1 = round(f1_score(y_test, y_pred), 4)
            accuracy = round(accuracy_score(y_test, y_pred), 4)
            precision = round(precision_score(y_test, y_pred), 4)
            recall = round(recall_score(y_test, y_pred), 4)
            cm = confusion_matrix(y_test, y_pred)
            processing_time = round(process_time() - start_time, 2)
            
            dict_results = {"Method" : [method], "Scaled": scaled, "Accuracy": accuracy,
                            "Precision" : [precision], "Recall" : [recall], "F1" : [f1],
                            "ConfusionMatrix" : str(cm), "ProcessingTime" : processing_time}
            df_results = df_results.append(pd.DataFrame(dict_results))
            
            if verbose:
                print(method.lower().ljust(20, " "), str(scaled).ljust(10, " "), str(accuracy).ljust(10, " "), str(f1).ljust(10, " "), processing_time)
    
    df_results = df_results.sort_values(by=('Accuracy'), ascending=False)
    df_results = df_results[["Method", "Scaled", "Accuracy", "F1", "Precision",
                             "Recall", "ConfusionMatrix", "ProcessingTime"]]

    if verbose:
        print("\nEND ML_LOOP\n" + "#" * 20)
        print("TOTAL CPU TIME:", round(process_time() - start, 2))
    return df_results


## TO CHECK
## Compression
#import seaborn as sns
#
#sparse_dataset = dataset
##dense_size = np.array(dataset).nbytes/1e6
##sparse_size = (sparse_dataset.data.nbytes + sparse_dataset.indptr.nbytes + sparse_dataset.indices.nbytes)/1e6
#
##sum(dataset.memory_usage())/ 1e6
#
#sns.barplot(['DENSE', 'SPARSE'], [dense_size, sparse_size])
#plt.ylabel('MB')
#plt.title('Compression')
    