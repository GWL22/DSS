# -*- coding: utf-8 -*-

import numpy as np
import math


class preprocess_dataset(object):
    def __init__(self, dataset):
        self.dataset = dataset

    # make list of NaT
    def find_nat(self, label):
        labeling = self.dataset[label]
        return self.dataset[labeling.isnull()].index

    # fill NaT in columns; 'srch_co', 'srch_ci'
    def fill_the_date(self, nat_list, label):
        u_id = self.dataset['user_id']
        labeling = self.dataset[label]
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
    def make_columns(self, label, resource1, resource2):
        self.dataset[label] = self.dataset[resource1] - self.dataset[resource2]
        self.dataset[label] = self.dataset[label] / np.timedelta64(1, 'D')

    # fill new columns if they have NaN
    def fill_columns(self, label, mean):
        labeling = self.dataset[label]
        u_id = self.dataset['user_id']
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

        labeling = labeling.astype(int)

    def make_sample(self):
        nat_list = self.find_nat('srch_ci')
        nat_list2 = self.find_nat('srch_co')
        self.make_columns('nights', 'srch_co', 'srch_ci')
        self.make_columns('prepare', 'srch_ci', 'date_time')
        print 'make columns'
        self.fill_the_date(nat_list, 'srch_co')
        self.fill_the_date(nat_list2, 'srch_ci')
        print 'fill NaT'
        self.fill_columns('nights', 1)
        self.fill_columns('prepare', 0)
        print 'fill columns'
        self.dataset = self.dataset.drop(['date_time',
                                          'site_name',
                                          'srch_ci',
                                          'srch_co'], axis=1) \
                                   .dropna()
        print 'complete'
        print self.dataset.head()
        return self.dataset

# test_line

# import dask.dataframe as dd
#
# num = 0
#
# whole_train = dd.read_csv('dataset/train.csv',
#                           parse_dates=['date_time',
#                                        'srch_ci',
#                                        'srch_co'])
# train_temp = whole_train.get_partition(num)
# pre_train = train_temp.head(len(train_temp))
# all_train = preprocess_dataset(pre_train).make_sample()
