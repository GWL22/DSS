# -*- coding: utf-8 -*-

import dask.dataframe as dd
import numpy as np

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from preprocessingdata import preprocess_dataset
from sklearn.cross_validation import train_test_split
from sklearn.ensemble \
    import RandomForestClassifier, BaggingClassifier, VotingClassifier

# Num is random choosed number in 0~121
num = 0

# train.csv is too big to hold all, Dask splits it into 122 partitions
whole_train = dd.read_csv('dataset/train.csv',
                          parse_dates=['date_time', 'srch_ci', 'srch_co'])
# Load one of partitions
train_temp = whole_train.get_partition(num)
# To make pandas, use .head()
pre_train = train_temp.head(len(train_temp))
# To preprocess, run make_sample() in preprocessingdata.py
all_train = preprocess_dataset(pre_train).make_sample()
# Memory may be hard to handle whole data, so all_train must be splited
# In this case, mix all_train and choose 10000 in that.
train = all_train.reset_index(np.random.permutation(all_train.index)) \
                 .head(10000)
train = train.set_index('index')

x = train.drop('hotel_cluster', axis=1)
y = train['hotel_cluster']

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.33,
                                                    random_state=0)
# BaggingClassifier has best score of accuracy_score : 0.33
model1 = BaggingClassifier(DecisionTreeClassifier(),
                           bootstrap_features=True,
                           random_state=0).fit(x_train, y_train)
print 'model BGC is ready'

# Support Vector machine using rbf kernel is 3rd : 0.29
# probability need to predict
model2 = SVC(kernel='rbf', probability=True).fit(x_train, y_train)
print 'model SVC is ready'

# RandomForestClassifier is 3rd : 0.3
model3 = RandomForestClassifier().fit(x_train, y_train)
print 'model RFC is ready'

# VotingClassifier is consist of several models
# To use weights, voting is 'soft'.
model4 = VotingClassifier(estimators=[('BGC', model1),
                                      ('SVC', model2),
                                      ('RFC', model3)],
                          voting='soft',
                          weights=[1, 1, 1]).fit(x_train, y_train)
print 'model VC is ready'

print '='*10
print 'Accuracy : {}'.format(accuracy_score(y_test, model4.predict(x_test)))
print '='*10
