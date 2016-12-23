# -*- coding: utf-8 -*-

import dask.dataframe as dd

from preprocessingdata import preprocess_dataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

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
train = preprocess_dataset(pre_train).make_sample()
# 5 columns need to analyze;
# user_location_country, hotel_country, srch_destination_id, nights, prepare
# delete columns excepte them
x = train.drop(['user_id',
                'orig_destination_distance',
                'user_location_city',
                'user_location_region',
                'posa_continent',
                'hotel_continent',
                'srch_adults_cnt',
                'srch_children_cnt',
                'srch_rm_cnt',
                'srch_destination_type_id',
                'is_booking',
                'cnt',
                'channel',
                'is_mobile',
                'hotel_market',
                'hotel_cluster',
                'is_package'], axis=1)
y = train['is_package']

# split train, test set
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.33,
                                                    random_state=0)

# Too much depth, Too much computation workload.
# To prevent above situation, depth should be optimized.
# In this case, accuracy_score is the best when max_depth is 100.
model = DecisionTreeClassifier(max_depth=100).fit(x_train, y_train)
print '='*10
print 'Accuracy : {}'.format(accuracy_score(y_test, model.predict(x_test)))
print '='*10
