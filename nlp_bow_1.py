# Natural Language Processing

# Importing usual libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from nlp_util import build_corpus, ml_loop


# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Build the corpus from reviews
corpus = build_corpus(dataset['Review'])

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray() # creates a sparse matrix

y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)



# Limit to specific algorithms
#methods = None #all methods
methods = ["logistic_regression", 
           "k-nn", 
           "naive_bayes", 
           "random_forest", 
           "svm_linear", 
           "svm_rbf", 
           "svm_sigmoid"]
# Feature scaling on that bag of words, not, or both?
scales = None

# Exemple limited to 3 methods without scaling
#methods = ["svm_linear", "svm_rbf", "svm_sigmoid"]
#scales = [False]



# Loop over all methods
df_results = ml_loop(X_train, y_train, X_test, y_test, methods=methods, scales=scales)

# Now have a look at the df_results dataframe to get the comparison
#TODO display a nice table with matplotlib - https://matplotlib.org/gallery/misc/table_demo.html#sphx-glr-gallery-misc-table-demo-py

