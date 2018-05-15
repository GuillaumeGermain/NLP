"""
Natural Language Processing
Bag or words model

looping over vocabulary size, training size
using classical classifier algorithms


"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from nlp_util import build_corpus, ml_loop

# Importing the datasets
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
dataset2 = pd.read_csv('newdata/1-restaurant-train.tsv', delimiter='\t', quoting=3)

#dataset.describe()
#dataset2.describe()


## The second dataset has different values, from 1 to 5
## middle values 3 will be deleted, 1-2 become 0 (Don't like), and 4-5 become 1 (Liked)
## Size 82000 -> 68000 (roughly)
dataset2.loc[dataset2['Liked'] < 3, 'Liked'] = 0
dataset2.loc[dataset2['Liked'] > 3, 'Liked'] = 1

# Dumb test
#dataset2.drop(dataset2[dataset2.Liked == 3].index, inplace=True)
dataset2.loc[dataset2['Liked'] == 3, 'Liked'] = 0

TEST_SIZE = len(dataset)




### LOOP OVER THIS
# Pre-cabled loops over many configurations to use directly

# Limit to specific algorithms
#methods = None # try all algos
methods=["logistic_regression"
         ,"k-nn"
         ,"naive_bayes"
         ,"random_forest"
         #,"svm_linear"
         #,"svm_rbf"
         #,"svm_sigmoid"
         #,"svm_poly"
         ]

methods=["logistic_regression"]


scales = None # try with both feature scaling and without

train_sizes = [15000, 18000]
vocab_sizes = [1800, 2200]
#train_sizes = [1000, 2000, 5000, 10000, 15000, 20000, 30000, len(dataset2)]
#vocab_sizes = [500, 800, 1000, 1500, 1800, 2000, 2200]


for train_size in train_sizes:
    for vocab_size in vocab_sizes:
        #train_size = 1000
        #vocab_size = 500
        
        dataset_work = dataset.append(dataset2[:train_size], ignore_index=True)
        corpus = build_corpus(dataset_work['Review'])
        
        # Creating the Bag of Words model for given train+test sets
        cv = CountVectorizer(max_features=vocab_size)
        X = cv.fit_transform(corpus).toarray()
        y = dataset_work.iloc[:, 1].values
        
        # set the matrix type to "sparse"
        #X = sparse.csr_matrix(X)
        
        # Get back train and test sets from merged matrix
        X_test = X[:TEST_SIZE]
        X_train = X[TEST_SIZE:TEST_SIZE + train_size]
        y_test = y[:TEST_SIZE]
        y_train = y[TEST_SIZE:TEST_SIZE + train_size]
        
        df_results = ml_loop(X_train, y_train, X_test, y_test, methods=methods, scales=scales)

