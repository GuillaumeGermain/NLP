# NLP Bag of Words
## Fun with traditional ML algorithms
The idea is to check how effective "traditional" ML methods work in a simple context.
1000 restaurant customer reviews are classified as 1 or 0, "Liked" or "Didn't like".

- Reviews are trimmed from most common words, the "stop words".
- Linked words/verbs (Like, Liked, Liking, Likable, etc.) are grouped together to get a more consistent result: "stemming".
- The resulting reviews are grouped into a rough "Bag of Words"
- This bag of words is processed with "traditional" ML techniques (with and without feature scaling), to predict ratings (0/1) on a test set of 200 reviews.
- Results are grouped in a table, sorted by descending accuracy. 

The most accurate algorithms in this case are:
1. SVM with sigmoid kernel and feature scaling
2. Logistic regression with feature scaling
3. SVM RBF with feature scaling

To review the results, open the df_results after the script has run.

Based on an exercise of ML A-Z on Udemy (Superdatascience)

## Phase 1:
### Algorithms:
* SVM (linear, sigmoid and RBF kernels)
* Naive Bayes
* K-NN
* logistic regression
### Surprising results
The SVM algorithm with Sigmoid kernel is both best and worst in class. With feature scaling, it achieves the best result with 79% accuracy. Without scaling, it's broken and always predicts 0 (Didn't like) values, as shown in the confusion matrix.
Some methods work better on a scaled input, others prefer the unscaled version.

![NLP Bag of Words Results](nlp_bow_1.png)

### Phase 1 Conclusions
Almost 80% accuracy after training on a sample of 800 reviews, is surprisingly good!! Specially with such a rough "bag of words" method, not taking into account the words order.
Now, that'd be nice to achieve a better accuracy, like 90%.

## Phase 2: Turbocharge the classifiers!
To achieve a better performance with my classifiers, my plan was to train them on a bigget dataset.
I found this one, with around 82000 reviews, on [Kaggle](https://www.kaggle.com/c/restaurant-reviews/data "Restaurant reviews")
As it turned out, it didn't work as expected, and I learned a few things on the way.

This second dataset had around 82000 lines, and scores between 1 and 5.
So I set 1-2 values to 0, and 4-5 to 1. 3 are discarded as they can't really be considered as "Liked" or not.

Doing so, it was quite an interesting exercise trying different configurations: vocabulary size (number of words considered) and training set sizes.

### Step by step execution
I first ran the classifiers with different training and vocabulary sizes to get a rough idea of the performance.
Results after training on 5000 observations of the second dataset, and testing on 1000 observations of the first one
![Results with training on second dataset](nlp_2_results.png)
```
####################
START ML_LOOP
training set size: 5000
vocabulary: 1000
test set size: 1000 

METHODS: logistic_regression k-nn naive_bayes random_forest svm_linear svm_rbf svm_sigmoid 
SCALING: [False, True] 

METHOD               SCALED     ACCURACY   F1         PROCESSING_TIME
logistic_regression  False      0.64       0.7293     0.31
k-nn                 False      0.54       0.2508     7.17
naive_bayes          False      0.518      0.6748     0.11
random_forest        False      0.531      0.6807     12.76
svm_linear           False      0.639      0.7267     12.17
svm_rbf              False      0.5        0.6667     13.71
svm_sigmoid          False      0.5        0.6667     13.91
logistic_regression  True       0.64       0.7273     2.4
k-nn                 True       0.51       0.6707     7.03
naive_bayes          True       0.518      0.6748     0.09
random_forest        True       0.53       0.6803     12.68
svm_linear           True       0.655      0.7324     12.22
svm_rbf              True       0.544      0.6864     16.41
svm_sigmoid          True       0.54       0.6845     10.27

END ML_LOOP
####################
TOTAL CPU TIME: 121.46
```

```
####################
START ML_LOOP
training set size: 10000
vocabulary: 1000
test set size: 1000 

METHODS: logistic_regression k-nn naive_bayes random_forest svm_linear svm_rbf svm_sigmoid 
SCALING: [False, True] 

METHOD               SCALED     ACCURACY   F1         PROCESSING_TIME
logistic_regression  False      0.671      0.7459     0.45
k-nn                 False      0.612      0.6381     14.49
naive_bayes          False      0.508      0.6702     0.2
random_forest        False      0.551      0.6901     30.22
svm_linear           False      0.673      0.7455     62.85
svm_rbf              False      0.5        0.6667     46.54
svm_sigmoid          False      0.5        0.6667     48.3
logistic_regression  True       0.673      0.7443     2.59
k-nn                 True       0.507      0.6698     14.23
naive_bayes          True       0.508      0.6702     0.15
random_forest        True       0.551      0.6901     29.56
svm_linear           True       0.678      0.7473     681.87
svm_rbf              True       0.588      0.7078     53.99
svm_sigmoid          True       0.592      0.7086     35.61

END ML_LOOP
####################
TOTAL CPU TIME: 1021.39

```
From this, SVM linear and logistic regression started to look like promising classifiers.
SVM linear was the best at this point, but its computation time was getting problematic, specially on the scaled input.
As it turned out, I dropped it later because it was way too long to process.

Naive Bayes turned out to be the fastest but really not accurate.
Random Forest (500 trees) was a bit long, and not so accurate either.

So I gave all of these classifiers a new try with a bigger training set and vocabulary size:

```
####################
START ML_LOOP
training set size: 15000
vocabulary: 1500
test set size: 1000 

METHODS: logistic_regression k-nn naive_bayes random_forest svm_linear svm_rbf svm_sigmoid 
SCALING: [False, True] 

METHOD               SCALED     ACCURACY   F1         PROCESSING_TIME
logistic_regression  False      0.683      0.7514     0.62
k-nn                 False      0.521      0.6657     29.5
naive_bayes          False      0.524      0.6775     0.53
random_forest        False      0.552      0.6906     69.43
svm_linear           False      0.698      0.7565     236.88
svm_rbf              False      0.5        0.6667     155.64
svm_sigmoid          False      0.5        0.6667     167.2
logistic_regression  True       0.683      0.7482     5.39
k-nn                 True       0.526      0.4        28.45
naive_bayes          True       0.524      0.6775     0.46
random_forest        True       0.552      0.6906     68.7
svm_linear           True       0.713      0.7626     2230.51
svm_rbf              True       0.63       0.7291     288.79
svm_sigmoid          True       0.607      0.7167     106.57

END ML_LOOP
####################
TOTAL CPU TIME: 3389.59```

At this point:
 
- K-NN accuracy is getting worse with more data
- Naive Bayes still has a minor improvement, but is super-fast then cheap to keep
- Random Forest is stagnating around 51-53% and bit long to process => out
it seems reasonable to drop K-NN, Naive Bayes and Random Forest (acceptably fast but not accurate)
The execution time of SVM linear was OK on a non-scaled input
SVM linear "scaled" seemed to deserve a special treatment as it had the best prediction, but it gets so long to train that it's not possible to keep it.
- SVM Sigmoid happens to stagnate around 50% so out.
- SVM RBF

## Lessons learned

### First mistakes!
Test on the right dataset!
I first trained and tested the classifiers on a mix of both datasets. This lead to amazingly good results, but it was not representative of the performance on the initial test set, what actually really mattered.

Second, I used the F1 score as main metric, accuracy turned out to be a better metric.

### Sparsity is a real problem
The matrix after tokenization is definitely very sparse. The matrix size is HUGE compared to the information it contains. The information density is very low.
This can lead to a longer computation time, as each cell has to be processed, whether it contains a dumb 0 or not.
Applying feature scaling changed the 0s by many mean values, but the information density was still the same (appalling).

### Computation Performance 
The matrix size is basically training set size * vocabulary size. So it could vary from 1000 x 1000 (1 million cells) to 50'000 x 18'000 (900 millions cells!) depending on encoding choices.
Scripts happened to run quite long. So I ensured that intermediate results got printed during the processing.
It became my main tool to evaluate results.

First, running with around 18K words and not too small datasets, it took around 25 minutes to process all the methods.
I also logged the times, which happened to be a very useful information.

It was quite interesting to compare the evolution of the time of specific classifiers, depending on the input size.
SVM RBF and SVM Sigmoid seemed to be quadratic o(n2), K-NN linear, logistic regression and naive Bayes lightning fast whatever the input.
I didn't plan to really improve the system performance, as I was mainly using standard parameters of these traditional algorithms.
Trying to writing them in Python was not an option (Python per se is awfully slow), and their implementation in C/C++ was definitively better than what I could have done myself.

### Evaluation metric
At first I used F1 score to select the best classifiers. 
As the 0-1 class distributions are quite balanced in both datasets (around 40-60%), the accuracy is actually the best metric to use. 

### Dataset quality and distribution
Finally, I compared the data distribution:
- Liked scores: the second dataset had a relatively higher mean (0.67 vs 0.5) and smaller standard deviation (0.3 vs 0.5)
Indeed, the initial dataset was quite extreme in its values, 0 or 1.
- The second dataset reviews were also much longer, and a lot more vocabulary (18000 vs 1800).
It also contained sentences not related to the restaurant itself.
E.g. "after a long day blah blah ... then we found that restaurant, then <real start of the review>"

Basically, I trained my classifiers to shoot in a target (training set), and the test set target was somewhere else. The datasets at the end were too different.

So the naive assumption that training ML classifiers on a bigger dataset would automatically bring better results, was indeed very naive.
Dataset quality matters!

## Phase 2 conclusions
Overall, I came to the conclusion that using this new dataset was not so great for training:
- In the best cases, both accuracy and F1 score very unsatisfying, specially taking into account the much higher training set size and longer processing time.
The performance was much better when training classifiers with only a part of the initial smaller dataset!
- One interesting fact, one of the most simple approaches, logistic regression, was always the fastest and generally with the best performance (most runs in a fraction of a second, 1000 times faster than others).
- Naive Bayes also had a decent result, and it was also fast compared to other algorithms. It makes sense to use it considering its performance.

## Phase 3: More research
Still somehow unsatisfied with these results, I wanted to find out if I could get more decent results, even though it was not the best one for training my classifiers.
I played a bit with the sparse matrix format for specific algorithms (except Naive Bayes which accepts only standard numpy matrices).
All in all, most of the best algorithms on the initial dataset didn't perform well when trained on the new dataset.

Logistic regression and SVM linear showed clearly the best results, depending on the training dataset and vocabulary sizes.
Logistic Regression is very interesting in this regards: by far the fastest, most of the time below a second, while others could take from 5-30 minutes to run, up to an eternity for SVM linear.

At the end I automated the tests with different settings (training set size and vocabulary size), to compare results.
So they can be called very quickly in a new context (new project or Hackathon). One case could be Spam Recognition, where Naive Bayes usually shines.

The results have been retrofitted in the previous parts of what now looks like a research blog.

### Asymptotic complexity:
Logistic regression is by far the most performant, the duration is most likely logarithmic-like (doubling the size barely increases the execution time).

SVM linear computation time is... very bad: from 10'000 to 20'000 lines, execution jumps from 25 to 321s.
This SVM linear provided by skikit seems to have beetween O(n3) and o(n4) asymptotic complexity.
I could not use it on even half of the whole second dataset because it was never completing.
It was a pity to discard it as it had regularly a better performance than logistic regression.
I'd still would like to find out, up to what point SVM linear still makes sense, and find its optimal performance.

Side remark: the execution is generally slower when performing over a scaled input.


# Final conclusions
The best results I could get was 74.2% with Logistic regression, with 35'000 observations, scaling and a vocabulary of 4500.
Overall, increasing the training dataset over a certain point seems to slightly decrease performance.
It suggests overfitting or dancing around a local optimum (as I would see it with a neural network and gradient descent).

- The winner in this specific context is Logistic Regression algorithm, though its performance is worse than in the first phase.
It is actually the second best, but by far the fastest.
SVM linear seems to be overall slightly better but its full potential cannot be used due to its awful computation time, which prevents training it on bigger datasets.
- All other algorithms, SVM RBF and SVM Sigmoid or the usually "hassle-free" Random Forests, and Naive Bayes, have a pretty bad performance, below 55%, maybe 60% at best with training/vocabulary sizes.
Overall not the best in this context.
- Feature scaling sometimes improves performance, sometimes not, depending on sizes.
I could find a kind of pattern with logistic regression, with scaling improving performance 2-3%... but not always.
So I keep both, though it significantly slows down the processing.
- At the end there is here a second level problem: running scripts over these 3 independent factors: vocabulary size, training size, feature scaling yes/no, and finding out if a pattern can be found.
Are there one or several local optima, an recommended size of vocabulary, where does feature scaling really help?
This would actually be very useful.
- The best performance was not as expected. On the point of view of the logistic regression, the accuracy actually increased from 64% to 74% after training on a new dataset.

## TODO
- Store all results from different settings in a single dataframe, including the configuration itself. 
- Test on EXACTLY the same test set of 200 observations, as when classifiers were trained only on 800 observations of the first dataset.
Just to be sure that this is not biasing the result.

## Next steps for improvement
I'll come back later on this exercise. Please send me feedbacks if you play a bit with it!

Basically, potential tracks for improvements would be:

1. Gather the result from a decently large sample of runs and find out the best settings, using a second level of ML algorithms (or why not a small NN). The most fun definitively.
2. Re-train on 800 of the first dataset + 5000-20'000 of the second one and check the results.
3. Find a relevant dataset with a good quality, and similar scale/distribution as the initial one.
4. Try with new classifiers
5. Of course the RNN NLP way: a simple recurrent neural network (RNN), next step with GRU cells.
6. Try different a fully different approach of bag of words: 2-grams or 3-grams would be promising
