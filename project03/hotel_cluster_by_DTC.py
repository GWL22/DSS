# -*- coding: utf-8 -*-

import pandas as pd
import dask.dataframe as dd
import numpy as np
import matplotlib.pyplot as pyplot
import math

from operator import itemgetter
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.cross_validation import train_test_split

whole_train = dd.read_csv('dataset/train.csv',
                          parse_dates=['date_time', 'srch_ci', 'srch_co'])


# make list of NaT
def find_nat(dataset, label):
    labeling = dataset[label]
    return dataset[labeling.isnull()].index


# fill NaT in columns; 'srch_co', 'srch_ci'
def fill_the_date(dataset, nat_list, label):
    u_id = dataset['user_id']
    labeling = dataset[label]
    for item in nat_list:
        flag_m = 0
        flag_p = 0
        for alpha in range(1, 100):
            if flag_m == 0:
                if (item-alpha) not in nat_list:
                    item1 = item - alpha
                    flag_m = 1
                else:
                    continue
            if flag_p == 0:
                if (item+alpha) not in nat_list:
                    item2 = item + alpha
                    flag_p = 1
                else:
                    continue
            elif flag_m + flag_p == 2:
                break

        if u_id.ix[item] == u_id.ix[item1]:
            labeling.ix[item] = labeling.ix[item1]
        elif u_id.ix[item] == u_id.ix[item2]:
            labeling.ix[item] = labeling.ix[item2]


# make new columns; nights, margin
def make_columns(dataset, label, resource1, resource2):
    dataset[label] = dataset[resource1] - dataset[resource2]
    dataset[label] = dataset[label] / np.timedelta64(1, 'D')


# fill new columns if they have NaN
def fill_columns(dataset, label, mean):
    labeling = dataset[label]
    u_id = dataset['user_id']
    for num in range(len(labeling)):
        if math.isnan(labeling.ix[num]) or labeling.ix[num] < 0:
            num1 = num - 1
            num2 = num + 1
            if u_id.ix[num] == u_id.ix[num1] and math.isnan(labeling.ix[num1]) is False:
                labeling.ix[num] = labeling.ix[num1]
            elif u_id.ix[num] == u_id.ix[num2] and math.isnan(labeling.ix[num2]) is False:
                labeling.ix[num] = labeling.ix[num2]
            else:
                labeling.ix[num] = mean

    labeling.astype(int)

# test line
num = 0

train_temp = whole_train.get_partition(num)
train = train_temp.head(len(train_temp)).dropna()
print 'ready to preprocess'

nat_list = find_nat(train, 'srch_ci')
nat_list2 = find_nat(train, 'srch_co')
make_columns(train, 'nights', 'srch_co', 'srch_ci')
make_columns(train, 'prepare', 'srch_ci', 'date_time')

print 'start to fill the data; NaT'
fill_the_date(train, nat_list, 'srch_co')
fill_the_date(train, nat_list2, 'srch_ci')
# the most nights user chosed is 1
fill_columns(train, 'nights', 1)
# the most margin user chosed is 0
fill_columns(train, 'prepare', 0)
print 'End to fill the data'

# For the memory, choose 20000 of data
train = train.reset_index(np.random.permutation(train.index)).head(10000)
train = train.set_index('index')
train['prepare'] = train['prepare'].astype(int)
train['nights'] = train['nights'].astype(int)
print 'make sample from train'

# delete the columns don't need to analyze
del train['srch_co']
del train['srch_ci']
del train['date_time']
del train['site_name']
# del train['user_id']
# del train['orig_destination_distance']

x = train.drop('hotel_cluster', axis=1)
y = train['hotel_cluster']


model1 = BaggingClassifier(DecisionTreeClassifier(),
                           bootstrap_features=True,
                           random_state=0).fit(x_train, y_train)
print 'model BGC is ready'

model2 = SVC(kernel='rbf', probability=True).fit(x_train, y_train)
print 'model SVC is ready'

model3 = RandomForestClassifier().fit(x_train, y_train)
print 'model RFC is ready'

model4 = VotingClassifier(estimators=[('BGC', model1),
                                      ('SVC', model2),
                                      ('RFC', model3)],
                          voting='soft',
                          weights=[1, 1, 1]).fit(x_train, y_train)
print 'model VC is ready'

# print classification_report(y_test, model_predict4)
print '='*10
print 'Accuracy : {}'.format(accuracy_score(y_test, model5.predict(x_test)))
print '='*10
