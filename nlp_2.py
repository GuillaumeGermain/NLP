"""
Natural Language Processing
Version 2 with MORE data


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


one test: svm linear and logistic regression have consistently the best results
naive bayes and svm sigmoid also were good on the original dataset
let's experiment with only these ones


Random Forest has consistently bad results and are pretty long to run
SVM linear here seems the most interesting: longest run time, but best performance
Naive Bayes which was best in the first dataset, has here only around 0.5 accuracy

!! naive Bayes is not able to manage big amounts -> partial_fit for "online fitting" or further fitting on new datase chunks
?? same for other methods?
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from nlp_util import *

# Importing the datasets
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
dataset2 = pd.read_csv('newdata/1-restaurant-train.tsv', delimiter='\t', quoting=3)

# Activate this to compare datasets
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



## The second dataset has different values, from 1 to 5
## middle values 3 will be deleted, 1-2 become 0 (Don't like), and 4-5 become 1 (Liked)
## Size 82000 -> 68000 (roughly)
dataset2.drop(dataset2[dataset2.Liked == 3].index, inplace=True)
dataset2.loc[dataset2['Liked'] < 3, 'Liked'] = 0
dataset2.loc[dataset2['Liked'] > 3, 'Liked'] = 1


# Compression

import seaborn as sns

sparse_dataset = dataset
#dense_size = np.array(dataset).nbytes/1e6
sparse_size = (sparse_dataset.data.nbytes + sparse_dataset.indptr.nbytes + sparse_dataset.indices.nbytes)/1e6

sum(dataset.memory_usage())/ 1e6

sns.barplot(['DENSE', 'SPARSE'], [dense_size, sparse_size])
plt.ylabel('MB')
plt.title('Compression')




# Append a part of the second biggest dataset to the main dataset
TEST_SIZE = len(dataset)
TRAIN_SIZE = 20000
VOCAB_SIZE = 1000


dataset2 = dataset2[:TRAIN_SIZE]
dataset = dataset.append(dataset2, ignore_index=True)


corpus = build_corpus(dataset['Review'])

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=VOCAB_SIZE)
X = cv.fit_transform(corpus).toarray() # creates a sparse matrix
y = dataset.iloc[:, 1].values

## Is X a sparse matrix? Well we already know that
#from scipy.sparse import csr_matrix, isspmatrix
#isspmatrix(csr_matrix([[5]]))



# Commented as we do it differently here
# the first 1000 lines of the dataset and hence X this is the test set
# the remaining of X is the training set.
# They are grouped together to set up the common vocabulary, cleaning and tokenization

## Splitting the dataset into the Training set and Test set
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Handle sparsity
from scipy import sparse
X = sparse.csr_matrix(X)



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





# Loop over methods

# Limit to specific algorithms
#methods = ["all"] # try all algos
methods = ["logistic_regression",
           #"naive_bayes",
           "svm_linear",
           #"svm_sigmoid"
           ]
#scales = [False, True]
#methods = ["svm_linear", "svm_rbf", "svm_sigmoid"]
#scales = [False]

df_results = ml_loop(X_train, y_train, X_test, y_test, methods=methods, scales=[False, True])
