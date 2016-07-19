# Identify Fraud From Enron Email

## Overview

In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives. In this project, I will play detective, building a person of interest identifier based on financial and email data made public as a result of the Enron scandal. This data have been combined with a hand-generated list of persons of interest in the fraud case, which means individuals who were indicted, reached a settlement or plea deal with the government, or testified in exchange for prosecution immunity.

This project uses the tools and starter code of the [Udacity's Intro to Machine Learning course](https://github.com/udacity/ud120-projects.git).

As preprocessing to this project, it has been combined the Enron email and financial data into a dictionary, where each key-value pair in the dictionary corresponds to one person. The dictionary key is the person's name, and the value is another dictionary, which contains the names of all the features and their values for that person. The features in the data fall into three major types, namely financial features, email features and POI labels.

financial features: ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] (all units are in US dollars)

email features: ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] (units are generally number of emails messages; notable exception is ‘email_address’, which is a text string)

POI label: [‘poi’] (boolean, represented as integer)

## Investigation process

### Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?

The goal of this project is try to find POI given some public financial data and emails from the well known Enron scandal. To accomplish the objetive I'm going to use the benefits of machine learning obtaining an algorithm that classify, given some training data, if a person is or not a POI. To create this algorithm I start with some training data. A small summary of it is:

```
Total number of datapoints:  146
Total number of features:  21
Summary features:  {'salary': 95, 'to_messages': 86, 'deferral_payments': 39, 'total_payments': 125, 'loan_advances': 4, 'bonus': 82, 'email_address': 111, 'restricted_stock_deferred': 18, 'total_stock_value': 126, 'shared_receipt_with_poi': 86, 'long_term_incentive': 66, 'exercised_stock_options': 102, 'from_messages': 86, 'other': 93, 'from_poi_to_this_person': 86, 'from_this_person_to_poi': 86, 'poi': 146, 'deferred_income': 49, 'expenses': 95, 'restricted_stock': 110, 'director_fees': 17}
Label class poi:  {False: 128, True: 18}
```

I can see an outlier plotting the features salary and bonus. There's a big difference with the next high value and obtaining the name of the datapoint that has those values, i see why these high values. It's the `TOTAL` aggregate of them so in this case i can remove the outlier because it don't have any value. The next high values are inside a 'logical' range so no more outliers are removed from the dataset.

### What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it.

#### Create new features

According to the course recommendations I create two new features:

* `fraction_from_poi`

    Is the fraction of emails that a person receive from POIs of the total received.

* `fraction_to_poi`

    Is the fraction of emails that a person send to POIs of the total sended.

Furthermore I created a new feature that is how much is the bonus according to his salary (bonus/salary).

* `fraction_bonus_salary`

#### Features selected

After new features creation and classification algorithm selection (Decision Tree Classifier), I used *feature importance* method to select and pick the features that will help me in the search of POIs

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

As we can see, two of the new features created (fraction_to_poi and fraction_from_poi) seems to be important in the search of POIs. The other ones won't give us any clue about them with 0 importance.

### What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?

Because we are handling a classification problem I tried with some of the most used algorithms (Naive Bayes, Support Vector Machine and Decision Tree) with resulting metrics:

```
GaussianNB()
	Accuracy: 0.73213	Precision: 0.23405	Recall: 0.44400	F1: 0.30652	F2: 0.37646
	Total predictions: 15000	True positives:  888	False positives: 2906	False negatives: 1112	True negatives: 10094

Got a divide by zero when trying out: SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
Precision or recall may be undefined due to a lack of true positive predicitons.
DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=None, splitter='best')
	Accuracy: 0.82467	Precision: 0.33729	Recall: 0.32650	F1: 0.33181	F2: 0.32860
	Total predictions: 15000	True positives:  653	False positives: 1283	False negatives: 1347	True negatives: 11717
```
So due to a better performance and the lack of information of SVM I ended up using **Decision Tree Classifier**.

### What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm?

The behaviour/results of an algorithm can be modified changing the value of the parameters. The idea is to tune the parameters of the algorithm to obtain better results. Each dataset is different and the algorithm parameters try to perform the best for each one.

To obtain the better values of the parameters of the Decision Tree Classifier I used GridSearchCV.

### What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?

### Give at least 2 evaluation metrics and your average performance for each of them. Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance.
