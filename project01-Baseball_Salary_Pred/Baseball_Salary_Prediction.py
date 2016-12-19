# -*- coding : utf-8 -*-

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
import seaborn as sns
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split

# Read the Batter's Data for 2015 Season
dataset = pd.read_csv('dataset/2015_Batter_Data.csv', sep=',')
dataset = dataset.drop(['ORDER', 'NAME', 'YEAR', 'P', 'TEAM'], axis=1)
# remove Pitcher and player who has not enough data
# There are only 4 cases satisfied with above conditions.
# although 222 data are decreased to 218, the result has effected too little
dataset = dataset.drop(dataset.index[[134, 164, 131, 132]])


class preprocessing_data(object):
    def __init__(self, dataset):
        self.dataset = dataset

    # scaling
    def dataset_scale(self):
        data_cols = list(self.dataset.columns)
        for col in data_cols[:-1]:
            self.dataset[col] = scale(self.dataset[col])
        return self.dataset

    # Divide Feautres and Y, want to predict.
    def make_xy(self):
        self.dataset = self.dataset_scale()
        dfX = self.dataset.drop(['SALARY'], axis=1)
        dfy = np.log(self.dataset['SALARY'])

        x_train, x_test, y_train, y_test = train_test_split(dfX, dfy,
                                                            test_size=0.2,
                                                            random_state=0)
        return x_train, x_test, y_train, y_test


class data_analysis(object):
    def __init__(self, dfX, dfy):
        self.dfX = dfX
        self.dfy = dfy

    # OLS Regression Report
    def show_OLS(self):
        dfx = sm.add_constant(self.dfX)
        model = sm.OLS(self.dfy, dfx)
        result = model.fit()
        print result.summary()

    def decision_PCA_num(self):
        pca = PCA().fit(self.dfX)
        var = pca.explained_variance_
        cmap = sns.color_palette()
        plt.bar(np.arange(1,len(var)+1), var/np.sum(var), align="center", color=cmap[0])
        plt.step(np.arange(1,len(var)+1), np.cumsum(var)/np.sum(var), where="mid", color=cmap[1])
        plt.show()
        result = 0
        count = 0
        for item in list(var):
            if result > 0.8:
                print 'Explain_ratio:' + str(result), 'PCA#:' + str(count)
                break

            else:
                result += item
                count += 1



    def PCA_modeling(self, num):
        X_pca = PCA(n_components=num)
        df_pca = X_pca.fit_transform(self.dfX)

        dfX_pca = pd.DataFrame(df_pca)
        dfX_pca.columns = ['PC1', 'PC2', 'PC3', 'PC4']

        df_pca_analysis = pd.concat([dfX_pca, self.dfy], axis=1)
        df_pca_analysis.columns = ['PC1', 'PC2', 'PC3', 'PC4', 'SALARY']

        regression = 'SALARY ~ PC1 + PC2 + PC3 + PC4'
        model_salary = sm.OLS.from_formula(regression, data=df_pca_analysis)
        result_salary = model_salary.fit()
        print result_salary.summary()

####################################################
dataset = pd.read_csv('dataset/2015_Batter_Data.csv', sep=',')
dataset = dataset.drop(['ORDER', 'NAME', 'YEAR', 'P', 'TEAM'], axis=1)
dataset = dataset.drop(dataset.index[[131, 132, 134, 164]])
data_cols = list(dataset.columns)
for col in data_cols[:-1]:
    if col == 'Foreigner':
        continue
    else:
        dataset[col] = scale(dataset[col])
dfX = dataset.drop(['SALARY'], axis=1)
dfy = np.log(dataset['SALARY'])
modeling1 = data_analysis(dfX, dfy)
modeling1.decision_PCA_num()
