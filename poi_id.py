#!/usr/bin/python

import sys
import pickle
sys.path.append("tools/")
import numpy as np

from feature_format import featureFormat, targetFeatureSplit
from summary import summary
from new_features import addFractionFeature
from tester import test_classifier, dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi', 'deferral_payments', 'total_payments', 'loan_advances', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees',  'fraction_from_poi', 'fraction_to_poi', 'shared_receipt_with_poi', 'fraction_bonus_salary'] # You will need to use more features
features_list = ['poi', 'total_payments', 'exercised_stock_options', 'other', 'fraction_to_poi', 'shared_receipt_with_poi', 'expenses', 'fraction_from_poi'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
summary(data_dict)

### Task 2: Remove outliers

data_dict.pop( 'TOTAL', 0 )

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
addFractionFeature(data_dict, "fraction_from_poi", "from_poi_to_this_person", "to_messages")
addFractionFeature(data_dict, "fraction_to_poi", "from_this_person_to_poi", "from_messages")
addFractionFeature(data_dict, "fraction_bonus_salary", "bonus", "salary")
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
"""
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features, labels)
test_classifier(clf, my_dataset, features_list)
"""
"""
from sklearn.svm import SVC
clf = SVC()
clf.fit(features, labels)
test_classifier(clf, my_dataset, features_list)
"""
from sklearn import tree
clf = tree.DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=4,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
            min_samples_split=1, min_weight_fraction_leaf=0.0,
            presort=False, random_state=42, splitter='best')
clf = clf.fit(features, labels)
test_classifier(clf, my_dataset, features_list)

###feature importances for feature selection process
"""
feature_importances = clf.feature_importances_
indices = np.argsort(feature_importances)[::-1]
print("Feature ranking:")
for f in range(len(indices)):
    print("%d. feature %s (%f)" % (f + 1, features_list[1:][indices[f]], feature_importances[indices[f]]))
"""
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
# from sklearn.cross_validation import train_test_split
# features_train, features_test, labels_train, labels_test = \
#     train_test_split(features, labels, test_size=0.1, random_state=42)
#     

from sklearn import grid_search

parameters = {
              'criterion': ['gini', 'entropy'],
              'splitter': ['best', 'random'],
              'max_depth': range(1,25),
              'min_samples_split':range(1,25),
              'random_state': [42]
              }
clf_p = grid_search.GridSearchCV(tree.DecisionTreeClassifier(), parameters).fit(features, labels)
print 'Parameters: ',clf_p.best_estimator_


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)