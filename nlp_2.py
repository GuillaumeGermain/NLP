# Natural Language Processing

# Version 2 with MORE data


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the datasets
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
dataset2 = pd.read_csv('newdata/1-restaurant-train.tsv', delimiter='\t', quoting=3)

# compare datasets
# set the same scoring scale for both
#dataset2.loc[dataset2['Liked'] == 1, 'Liked'] = 0.
#dataset2.loc[dataset2['Liked'] == 2, 'Liked'] = 0.25
#dataset2.loc[dataset2['Liked'] == 3, 'Liked'] = 0.5
#dataset2.loc[dataset2['Liked'] == 4, 'Liked'] = 0.75
#dataset2.loc[dataset2['Liked'] == 5, 'Liked'] = 1.
#
#dataset.describe()
#dataset2.describe()
#

"""
CONCLUSION

After quite some exercise (very good practice with Python and Pandas, data processing)
the new dataset used for training is too far from the test dataset.
It could have been expected that more data would help, but it was not the case

Surprisingly, the initial test taking into account the new dataset for testing, had very good results
It could be interesting to actually use these ML algos on this new dataset...

second, limiting the corpus to 4000 words may have noticeably deteriorated the performance
It is very curious here that a simple logistic regression had the shortest fitting time and the best performance

This has been confirmed by checking the distribution of the data
 datasets have a significantly different distribution
 the second dataset has a noticeably higher mean (0.67 vs 0.5)
 specially the standard deviation is VERY different (0.3 vs 0.5)
 this could explain why it is not useful to increase the performance

"""



## The second dataset has different values, from 1 to 5
## middle values 3 will be deleted, 1-2 become 0 (Don't like), and 4-5 become 1 (Liked)
## Size 82000 -> 68000 (roughly)
dataset2.drop(dataset2[dataset2.Liked == 3].index, inplace=True)
#dataset2.drop(dataset2[dataset2.Liked == 2].index, inplace=True)
#dataset2.drop(dataset2[dataset2.Liked == 4].index, inplace=True)
dataset2.loc[dataset2['Liked'] < 3, 'Liked'] = 0
dataset2.loc[dataset2['Liked'] > 3, 'Liked'] = 1
#dataset2.loc[dataset2['Liked'] == 3, 'Liked'] = 0.5




# Append a part of the second biggest dataset to the main dataset
TEST_SIZE = len(dataset)
TRAIN_SIZE = 5000
VOCAB_SIZE = 1000


dataset2 = dataset2[:TRAIN_SIZE]
dataset = dataset.append(dataset2, ignore_index=True)



# Build the corpus
def build_corpus(dataset):
    import re
    import nltk
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
    for i in range(len(dataset)):
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
        review = review.replace('\\n', ' ')
        review = review.lower().split()
        review = [ps.stem(word) for word in review if word not in en_stop_words]
        review = ' '.join(review)
        corpus.append(review)
    return corpus

# Process main dataset
corpus = build_corpus(dataset)


# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=VOCAB_SIZE)
X = cv.fit_transform(corpus).toarray() # creates a sparse matrix

## Is X a sparse matrix? Well we already know that
#from scipy.sparse import csr_matrix, isspmatrix
#isspmatrix(csr_matrix([[5]]))

y = dataset.iloc[:, 1].values


# Commented as we do it differently here
# the first 1000 lines of the dataset and hence X this is the test set
# the remaining of X is the training set.
# They are grouped together to set up the common vocabulary, cleaning and tokenization

## Splitting the dataset into the Training set and Test set
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)





# Loop over methods

## Limit to specific algorithms
#methods = ["logistic_regression",
#           "k-nn",
#           "naive_bayes",
#           "random_forest",
#           "svm_linear",
#           "svm_rbf",
#           "svm_sigmoid"]
#scales = [False, True]
#methods = ["svm_linear", "svm_rbf", "svm_sigmoid"]
#scales = [False]


def ml_loop(X_train, y_train, X_test, y_test, methods=["all"], scales=[False, True], verbose=True):
    """
    Parameters:
    X_train, y_train, X_test, y_test: training and test sets
    methods: algorithms used. by default "all", all those defined in this function
    scales: feature scaling on the matrix or not. the False value (non-scaled) should be first
    verbose: display intermediate results for longer datasets/vocabularies, which may take 30 minutes to run
    """
    from time import process_time
    start = process_time()
    
    if verbose:
        print("#" * 20 + "\nSTART ML_LOOP")
        print("training set size:", X_train.shape[0])
        print("test set size:", X_test.shape[0])
        print("vocabulary:", X_train.shape[1], "\n")
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
    
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
            # Feature Scaling
            from sklearn.preprocessing import StandardScaler
            sc_X = StandardScaler()
            X_train = sc_X.fit_transform(X_train.astype(float))
            X_test = sc_X.transform(X_test.astype(float))
            if verbose:
                print("\nSCALING DONE")
        
        for method in methods:
            start_time = process_time()
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



# Split X and Y to get back values of dataset and dataset2
# dataset will be used for testing, dataset2 for training
# X and y contain first dataset values, then dataset2 values
X_train = X[TEST_SIZE:TEST_SIZE + TRAIN_SIZE]
X_test = X[:TEST_SIZE]
y_train = y[TEST_SIZE:TEST_SIZE + TRAIN_SIZE]
y_test = y[:TEST_SIZE]

# manually trim if needed
#X_train = X_train[:500]
#y_train = y_train[:500]

df_results = ml_loop(X_train, y_train, X_test, y_test, methods=["all"], scales=[False, True])
