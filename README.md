# NLP Bag of Words
## Fun with traditional ML algorithms

1000 restaurant customer reviews are classified as 1 or 0, "Liked" or "Didn't like".
Reviews are trimmed, removing most common words, and common words/verbs (Like, Liked, Liking) are grouped together to get a more consistent result ("stemming")
The result is grouped into a rough "Bag of Words", and processed with "traditional" ML techniques to predict ratings on a test set of 200 reviews.
Results are grouped in a table, sorted by descending F1 score. 

The idea is to check how effective "traditional" ML methods work in this context.

The most effective algorithms in this case are SVM with a sigmoid kernel with feature scaling, then Naive Bayes without scaling
To review the results, open the df_results after the script has run.

## Algorithms:
* SVM (linear, sigmoid and RBF kernels)
* Naive Bayes
* K-NN
* logistic regression

Based on an exercise of ML A-Z on Udemy (Superdatascience)

## Surprising results
The SVM algorithm with RBF kernel is both best and worst in class. With feature scaling, it achieves the best result with 79% accuracy. Without scaling, it's broken and always predicts 0 (Didn't like) values, as shown in the confusion matrix.
Some methods work better on a scaled matrix, other prefer the unscaled version.

![NLP Bag of Words Results](nlp_bag_of_words_results.png)

## Phase 1 Conclusions
Almost 80% accuracy after training on a sample of 800 reviews, is surprisingly good!! Specially with such a rough "bag of words", not taking into account the order of words.
Now, that'd be nice to achieve a better accuracy, like 90%.

## Phase 2: Next step!
To achieve a better performance with my classifiers, my plan was to turbocharge them by training them on a bigget dataset of 82000 reviews, like this one on [Kaggle](https://www.kaggle.com/c/restaurant-reviews/data "Restaurant reviews")
As it turned out, it was not a great idea, but I learned a lot on the way.

The second dataset had roughly 82000 lines, and scores between 1 and 5.
So I set 1-2 values to 0, and 4-5 to 1. 3 are discarded as they can't really be considered as "Liked" or not.

Doing so, it was quite an interesting exercise trying different configurations: vocabulary size (number of words considered) and training set sizes.

### Typical results
Results after training on 5000 observations of the second dataset, and testing on 1000 observations of the first one
![Results with training on second dataset](nlp_2_results.png)
```
####################
START ML_LOOP
training set size: 5000
test set size: 1000
vocabulary: 1000 

METHODS: logistic_regression k-nn naive_bayes random_forest svm_linear svm_rbf svm_sigmoid 
SCALING: [False, True] 

METHOD               SCALED     F1         ACCURACY   PROCESSING_TIME
logistic_regression  False      0.7293     0.64       0.4
k-nn                 False      0.2508     0.54       7.58
naive_bayes          False      0.6748     0.518      0.13
random_forest        False      0.6807     0.531      12.76
svm_linear           False      0.7267     0.639      12.21
svm_rbf              False      0.6667     0.5        13.73
svm_sigmoid          False      0.6667     0.5        13.7

SCALING DONE
logistic_regression  True       0.7273     0.64       2.39
k-nn                 True       0.6707     0.51       7.45
naive_bayes          True       0.6748     0.518      0.1
random_forest        True       0.6803     0.53       12.68
svm_linear           True       0.7324     0.655      12.45
svm_rbf              True       0.6864     0.544      16.68
svm_sigmoid          True       0.6845     0.54       10.29

END ML_LOOP
####################
TOTAL CPU TIME: 122.8
```
## Lessons learned

### Test on the right dataset!
My initial mistake was to train and test the classifiers on a mix of both datasets. This lead to very surprisingly good results, but it was not representative of the performance on the initial test set, what actually really mattered.

### Sparsity is a real problem
The matrix after tokenization is definitely very sparse. The matrix size is HUGE compared to the information it contains. The information density is very low.
This is a performance problem, as each cell has to be processed, whether it contains a dumb 0 or not.
Applying feature scaling changed the 0s by many mean values, but the information density was still the same (appalling).

### System Performance 
The matrix size is basically training set size * vocabulary size. So it could vary from 1000 x 1000 (1 million cells) to 50'000 x 18'000 (900 millions cells!) depending on encoding choices.
Scripts happened to run quite long. So I ensured that intermediate results got printed during the processing.
It became my main tool to evaluate results.

First, running with around 18K words and not too small datasets, it took around 25 minutes to process all the methods.
I also logged the times, which happened to be a very useful information.

Unfortunately, I could not really improve the system performance, as I was mainly using standard fitting functions of traditional algorithms.
Trying to writing them in Python was not an option (Python per se is awfully slow), and their implementation in C/C++ was definitively better than what I could have done myself.

## Phase 2 conclusions
I first ran the classifiers with different training and vocabulary sizes to get a rough idea of the performance.
Overall, I came to the conclusion that using this new dataset was not so great for training:
- In the best cases, the F1 score was quite average, and the accuracy very unsatisfying, specially taking into account the much higher training set size and longer processing time.
The performance was much better when training classifiers with only a part of the initial smaller dataset!
- One interesting fact, one of the most simple approaches, logistic regression, was always the fastest and generally with the best performance (most runs in a fraction of a second, 1000 times faster than others).
- Naive Bayes also had a decent result, and it was also fast compared to other algorithms. It makes sense to use it considering its performance.

### Dataset quality
Finally, I compared the data distribution:
- Liked scores: the fist dataset had a relatively higher mean (0.67 vs 0.5) and smaller standard deviation (0.3 vs 0.5)
Indeed, the initial dataset was quite extreme in its values, 0 or 1.
- The second dataset reviews were much longer with a lot more vocabulary. It also contained elements not related to the restaurant itself. E.g. "after a long day blah blah ... then we found that restaurant, then <real start of the review>"

My initial idea came from transfer learning in the area of neural networks (e.g. CNN for Computer Vision). This is when a neural network is trained on a huge dataset, then is adapted to a very specific problem with a much smaller dataset.
But the reality is that I trained my classifiers to shoot in a target, and the test target was somewhere else. The datasets at the end were too different.

So the naive assumption that training ML classifiers on bigger dataset would automatically bring better results, was indeed very naive.
Dataset quality matters!

### Test evaluation
At first I used F1 score to select the best classifiers. As the 0-1 class distributions are quite balanced in both datasets (around 40-60%), the accuracy is actually the good metric to use. This is the one which is from now on used to evaluate the performance.

## Phase 3
Still a bit unsatisfied with the results, I tried a few things and cleaned up the scripts. 
I played a bit with the sparse matrix format for specific algorithms (except Naive Bayes which accepts only standard numpy matrices).
4 algorithms have been long-listed, from which 2 algorithms have been retained:
- naive_bayes, svm_sigmoid due to their initial good performance on the original dataset
- logistic_regression, svm_linear as they showed quite some correct performance after training on the second bigger dataset.

All in all, the best algorithms on the initial dataset, didn't perform well when trained on the new dataset.

Logistic regression and SVM linear showed the best results, depending on the training dataset and vocabulary sizes. Logistic Regression is very interesting in this regards: by far the fastest, most of the time below a second, while others could take from 5-30,  minutes to run, up to an eternity for SVM linear.

At the end I automated the tests with different settings (training set size and vocabulary size), to compare results.
So they can be called very quickly in a new context (new project or Hackathon). One case could be Spam Recognition, where Naive Bayes usually shines.

### Asymptotic complexity:
Logistic regression is by far the most performant, the duration is most likely logarithmic-like (doubling the size barely increases the execution time).
SVM linear is very bad: from 10'000 to 20'000 lines, execution changes from 25 to 321s. This SVM linear provided by skikit seems to have beetween O(n3) and o(n4) asymptotic complexity. I could not deploy on the whole second dataset and a larger vocabulary because it simply was running too long. It was a pity to discard it as it had regularly a better performance than logistic regression.
Side remark: the execution is generally slower when performing over a scaled input.

# Final conclusions
- The winner in this context is Logistic Regression algorithm, though its performance was not as expected. It was simply the second best, and very fast. SVM linear was overall slightly better but not an option due to its awful computation time.
- The usual "hassle-free" Random Forests, and Naive Bayes, had a pretty bad performance, around 50%.
- Feature scaling sometimes improves performance, sometime not.

## TODO
- store all results from different settings in a single dataframe, including the configuration itself. 
- clean up the code after a lot of small changes.
- test on EXACTLY the same test set of 200 observations, as when classifiers were trained only on 800 observations of the first dataset. Just to be sure that this is not biasing the result.

## Next steps for improvement
I won't go further on this exercise. Please send me feedbacks if you do!

Basically, potential tracks for improvements would be:

1. Gather the result from a large sampling of runs and find out the best settings, using a second level of ML algorithms (or why not a small NN). The most fun definitively.
2. Re-train on 800 of the first dataset + 5000-20'000 of the second one and check the results.
3. Find a relevant dataset with a good quality, and similar scale/distribution as the initial one.
4. Try different a fully different approach of bag of words: 2-grams or 3-grams would be promising
5. Of course the RNN NLP way: a simple recurrent neural network (RNN), next step with GRU cells.
