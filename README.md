# NLP Bag of Words
## Fun with traditional ML algorithms

1000 restaurant customer reviews are classified as 1 or 0, "Liked" or "Didn't like".
Reviews are trimmed, removing most common words, and common words/verbs (Like, Liked, Liking) are grouped together to get a more consistent result ("stemming")
The result is grouped into a rough "Bag of Words", and processed with "traditional" ML techniques to predict ratings on a test set of 200 reviews.
Results are grouped in a table, sorted by descending F1 score. 

The most effective algorithms in this case are SVM with a sigmoid kernel with feature scaling, then Naive Bayes without scaling
To review the results, open the df_results after the script has run

## Algorithms:
* SVM (Kernels linear, sigmoid, rbf)
* Naive Bayes
* K-NN
* logistic regression

Based on an exercise of ML A-Z on Udemy (Superdatascience)

![NLP Bag of Words Results](nlp_bag_of_words_results.png)
