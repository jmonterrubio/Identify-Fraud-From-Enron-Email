# Identify Fraud From Enron Email

## Overview

In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives. In this project, I will play detective, building a person of interest identifier based on financial and email data made public as a result of the Enron scandal. This data have been combined with a hand-generated list of persons of interest in the fraud case, which means individuals who were indicted, reached a settlement or plea deal with the government, or testified in exchange for prosecution immunity.

This project uses the tools and starter code of the [Udacity's Intro to Machine Learning course](https://github.com/udacity/ud120-projects.git).

As preprocessing to this project, it has been combined the Enron email and financial data into a dictionary, where each key-value pair in the dictionary corresponds to one person. The dictionary key is the person's name, and the value is another dictionary, which contains the names of all the features and their values for that person. The features in the data fall into three major types, namely financial features, email features and POI labels.

financial features: ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] (all units are in US dollars)

email features: ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] (units are generally number of emails messages; notable exception is ‘email_address’, which is a text string)

POI label: [‘poi’] (boolean, represented as integer)

## Investigation process

1. Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it.

  The goal of this project is try to find POIs given some public financial data and emails from the well known Enron scandal. To accomplish the objetive I'm going to use the benefits of machine learning obtaining an algorithm that classify, given some training data, if a someone is or not a POI. To create this algorithm I start with some training data. There are 146 datapoints and 21 features. The label class is `poi` and I can see how unbalanced is the dataset with 128 non-POI values against just 18 POI. Not all the datapoints have all the features filled with values, and features like `loan_advances` and `director_fees` with 142 and 129 NaN values respectively seems they're not going to help up very much in the investigation process.

  I can see an outlier plotting the features `salary` and `bonus`. The key behind those extreme values is `TOTAL` and consist of the aggregate of the values of each feature so in this case I can remove the outlier because it won't give us any benefit. The next high values are inside a 'logical' range so no more outliers are removed from the dataset.

2. What features did you end up using in your POI identifier, and what selection process did you use to pick them?

  1. Create new features

    According to the course recommendations I create two new features. They're created because it make sense that POIs send and receive more messages between them:

    * `fraction_from_poi`

        Is the fraction of emails that a person receive from POIs of the total received.

    * `fraction_to_poi`

        Is the fraction of emails that a person send to POIs of the total sended.

    Furthermore I created a new feature that is how much is the bonus according to his salary (bonus/salary ratio) that intuitively seems it may help in identifying POIs.

    * `fraction_bonus_salary`

  2. Features selected

    After new features creation and classification algorithm selection (Decision Tree Classifier), I used *feature importance* method to select and pick the features that will help me in the search of POIs:

    `exercised_stock_options`, `total_payments`, `fraction_to_poi`, `shared_receipt_with_poi`, `other`, `restricted_stock`, `fraction_from_poi`

    ```
    1. feature exercised_stock_options (0.331378)
    2. feature total_payments (0.142832)
    3. feature fraction_to_poi (0.127535)
    4. feature shared_receipt_with_poi (0.125238)
    5. feature other (0.122483)
    6. feature restricted_stock (0.080004)
    7. feature fraction_from_poi (0.070529)
    ```

    As we can see, two of the new features created (fraction_to_poi and fraction_from_poi) seems to be important in the search of POIs. The other ones that aren't in the list above won't give us any clue about them with 0 importance.

3.  What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?

  Because we are handling a classification problem I tried with some of the most used algorithms (Naive Bayes, Support Vector Machine and Decision Tree) with resulting metrics:

  ```
  GaussianNB()
  	Accuracy: 0.73213	Precision: 0.23405	Recall: 0.44400	F1: 0.30652	F2: 0.37646
  	Total predictions: 15000	True positives:  888	False positives: 2906	False negatives: 1112	True negatives: 10094

  Pipeline(steps=[('scaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('classifier', SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False))])
    	Accuracy: 0.86747	Precision: 0.58333	Recall: 0.02100	F1: 0.04054	F2: 0.02602
    	Total predictions: 15000	True positives:   42	False positives:   30	False negatives: 1958	True negatives: 12970

  DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
              max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
              min_samples_split=2, min_weight_fraction_leaf=0.0,
              presort=False, random_state=None, splitter='best')
  	Accuracy: 0.82467	Precision: 0.33729	Recall: 0.32650	F1: 0.33181	F2: 0.32860
  	Total predictions: 15000	True positives:  653	False positives: 1283	False negatives: 1347	True negatives: 11717
  ```
  So due to a better performance in this case (see the difference in Accuracy and Precission/Recall tradeoff) I ended up using **Decision Tree Classifier**.

4. What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm?

  The behaviour/results of an algorithm can be modified changing the value of the parameters. The idea is to tune the parameters of the algorithm to obtain better results. Each dataset is different and the algorithm parameters try to perform the best for each one.

  To obtain the better values of the parameters of the Decision Tree Classifier I used GridSearchCV obtaining some parameters for the Decision Tree Clasiffier that increase all the metrics used to qualify the algorithm.

5. What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?

  If you test a model on same data as used for training the learner, the model may appear to make overly accurate predictions. This is an example of overfitting. To reliably estimate the predictive power of a model, it should be tested on data that hasn't been used for training the learner. Cross validation lets you use all examples for both learning and testing without ever using the same sample for both training and testing.

  The way I'm validating the analysis is with the function `test_classifier` defined in the `tester.py`  script provided by the tools folder of the starter code. This script uses *stratified shuffle split cross validation*.

6. Give at least 2 evaluation metrics and your average performance for each of them. Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance.

  The metrics used to evaluate the performance of the algorithm are **Precision** and **Recall**. In our problem these metrics are better than accuracy to evaluate the performance because of the content of the dataset. We have a very skewed classes (labels) and the accuracy metric is not the ideal one. The final results of the metrics are:

  Metric | Result
  --- | ---
  Accuracy | 0.87600
  `Precision` | **0.54113**
  `Recall` | **0.46050**
  F1 | 0.49757
  F2 | 0.47464
  Total predictions | 15000
  True positives | 921
  False positives | 781
  False negatives | 1079
  True negatives | 12219

  In this case, the values of precision and recall means that a 46% of the real POIs are detected and a 54% of the detected POIs is really true.

## Bibliography

  * Course materials from the [Intro to Machine Learning](https://www.udacity.com/course/intro-to-machine-learning--ud120) Udacity course.
  * [Scikit learn documentation](http://scikit-learn.org/stable/)
  * Cross-Validation Info (https://rapid-i.com/wiki/index.php?title=Cross-validation)
