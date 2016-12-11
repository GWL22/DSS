# -*- coding: urf-8 -*-

import dask.dataframe as dd

from preprocessingdata import preprocess_dataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

whole_train = dd.read_csv('dataset/train.csv',
                          parse_dates=['date_time', 'srch_ci', 'srch_co'])

# random_number : num
num = 0

train_temp = whole_train.get_partition(num)
pre_train = train_temp.head(len(train_temp))

train = preprocess_dataset(pre_train).make_sample()

x = train.drop(['orig_destination_distance',
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

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.33,
                                                    random_state=0)

model = DecisionTreeClassifier(max_depth=100).fit(x_train, y_train)
print accuracy_score(y_test, model.predict(x_test))
