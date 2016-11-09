
# coding: utf-8

# In[1]:

# -*- coding : utf-8 -*-

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
import seaborn as sns
from sklearn.preprocessing import scale, robust_scale, minmax_scale, maxabs_scale
from sklearn.decomposition import PCA


# In[11]:

k1 = 'TASUK,TASU,DEUKJUM,ANTA,2TA,3TA,HOMERUN,ALLTA,TAJUM,DORU,DOSIL,BALLNET,SAGU,GOSA,SAMJIN,BYUNGSAL,HEUITA,HEUIBI,TAYUL,CHULRU,JANGTA,OPS,WOBA,WRC,WAR,WPA,CAREER,SALARY'
k2 = 'TASUK,TASU,DEUKJUM,ANTA,2TA,3TA,HOMERUN,ALLTA,TAJUM,DORU,DOSIL,BALLNET,SAGU,GOSA,SAMJIN,BYUNGSAL,HEUITA,TAYUL,CHULRU,JANGTA,OPS,WOBA,WRC,WAR,WPA'

a = list(k1.split(','))
b = list(k2.split(','))

#1년치 타자 데이터
df1 = pd.read_csv('salary_prediction_taja_add_career.csv')

df1.head()
dfX = pd.DataFrame(df1, columns = a[:-7])
dfY = pd.DataFrame(df1, columns = ['SALARY'])
dfZ = pd.DataFrame(df1, columns = ['CAREER'])

#희생타 = 희생번트 + 희생플라이
dfX['HEUITA'] += dfX['HEUIBI']
del dfX['HEUIBI']

#dfX

result = []
#타자 능력치 표준화 작업, Salary는 미포함!
for i in f:
    result.append(normalize(dfX[i]))

#한 선수당 1 column인 data table    
#db_batter_index = pd.DataFrame(np.vstack(result), index = d)
db_batter_index = pd.DataFrame(np.vstack(result), index = f)

#X 변수끼리의 상관계수를 계산하기 위해서는 index에 변수가 들어가야함.
db_batter_index
dfX = db_batter_index.T
#dfX.ix[0].T
db_batter_index


# In[12]:


# 상관행렬 구하기

#dataA = np.corrcoef(db_batter_index)
#cor_temp=[]
#for i in range(len(dataA)):
#    cor_temp.append(dataA[i].T)
    
#cor_matrix = pd.DataFrame(np.vstack(cor_temp), index = b, columns = b)
#cor_matrix


# In[13]:

X_pca = PCA(n_components=3)
X_pca.fit_transform(dfX)

# PCA 성분 개수

#g1 = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'CAREER', 'SALARY']
g1 = ['PC1', 'PC2', 'PC3', 'PC4']
dfX_pca = pd.DataFrame(X_pca.fit_transform(dfX))
dfX_pca.columns = ['PC1', 'PC2', 'PC3']
#dfX_pca


df_pca_analysis = pd.concat([dfX_pca, pd.DataFrame(normalize(dfZ['CAREER']).T), np.log(dfY)], axis = 1)
df_pca_analysis.columns = ['PC1', 'PC2', 'PC3', 'CAREER', 'SALARY']
df_pca_analysis
#dfX_pca


# In[14]:

X_pca = PCA(n_components=3)
X_pca.fit_transform(dfX)


# PCA 성분 개수

#g1 = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'CAREER', 'SALARY']
g1 = ['PC1', 'PC2', 'PC3', 'PC4']
dfX_pca = pd.DataFrame(X_pca.fit_transform(dfX))
dfX_pca.columns = ['PC1', 'PC2', 'PC3']
#dfX_pca


df_pca_analysis = pd.concat([dfX_pca, pd.DataFrame(robust_scale(dfZ)), np.log(dfY)], axis = 1)
df_pca_analysis.columns = ['PC1', 'PC2', 'PC3', 'CAREER', 'SALARY']
df_pca_analysis
#dfX_pca


# In[15]:

kp = pd.DataFrame(X_pca.components_.T, columns = ['PC1', 'PC2', 'PC3'], index = f)
kp


# In[16]:

#regression = 'Salary ~ PC1_Score + np.log(PC3_Score) + Career'
#regression = 'Salary ~ I(PC1_Score ** 2) + PC2_Score + PC4_Score + Career'
regression = 'SALARY ~ PC1 + PC2 + I(PC3 ** 2) + CAREER'
#regression = 'SALARY ~ PC1  + (np.sqrt(abs(PC2**2 - 0.2*PC3**2))) + CAREER'
#regression = 'SALARY ~ PC1'
model_salary = sm.OLS.from_formula(regression, data = df_pca_analysis)
result_salary = model_salary.fit()
print (result_salary.summary())


# In[17]:



#wine = fetch_mldata("wine")
#X, y = wine.data, wine.target

pca = PCA().fit(dfX)
var = pca.explained_variance_
cmap = sns.color_palette()

plt.bar(np.arange(1,len(var)+1), var/np.sum(var), align="center", color=cmap[0])
plt.step(np.arange(1,len(var)+1), np.cumsum(var)/np.sum(var), where="mid", color=cmap[1])
plt.show()


# In[18]:

print (pca.explained_variance_ratio_)


# In[19]:

influence = result_salary.get_influence()
hat = influence.hat_matrix_diag
plt.stem(hat)
plt.axis([ -2, len(dfY)+2, 0, 0.2 ])
plt.show()
print("hat.sum() =", hat.sum())


# In[20]:

plt.figure(figsize=(10, 2))
plt.stem(result_salary.resid)
plt.xlim([-2, len(dfY)+2])
plt.show()


# In[21]:

sm.graphics.plot_leverage_resid2(result_salary)
plt.show()


# In[22]:

sm.graphics.influence_plot(result_salary, plot_alpha=0.3)
plt.show()


# In[23]:

plt.figure(figsize=(10, 2))
plt.stem(result_salary.outlier_test().ix[:, -1])
plt.show()


# In[34]:

df_pca_analysis2 = df_pca_analysis.drop(36)

df_pca_analysis2


# In[35]:

df_pca_analysis3 = df_pca_analysis2.drop(20)

df_pca_analysis3


# In[36]:

regression = 'SALARY ~ PC1 + PC2 + I(PC3 ** 2) + CAREER'
model_salary = sm.OLS.from_formula(regression, data = df_pca_analysis3)
result_salary = model_salary.fit()
print (result_salary.summary())


# In[37]:

#regression = 'SALARY ~ PC1  + (np.sqrt(abs(PC2**2 - 0.2*PC3**2))) + CAREER'
#df_pca_analysis3['PC2'] = np.sqrt(abs((df_pca_analysis3['PC2'] ** 2) - (0.2 * df_pca_analysis3['PC3'] ** 2)))
df_pca_analysis3['PC3'] = (df_pca_analysis3['PC3'] ** 2)

df_pca_analysis3


# In[38]:

from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='median', axis=0)
pca_analysis4 = imp.fit_transform(df_pca_analysis3)

df_pca_analysis4 = pd.DataFrame(pca_analysis4, columns = ['PC1', 'PC2', 'PC3', 'CAREER', 'SALARY'])

df_pca_analysis4


# In[28]:

del df_pca_analysis3['PC3']

df_pca_analysis3


# In[39]:

sm.graphics.influence_plot(result_salary, plot_alpha=0.3)
plt.show()


# In[40]:

sns.pairplot(df_pca_analysis3)
plt.show()


# In[41]:

df_pca_analysis4_cv_x = pd.DataFrame(df_pca_analysis4, columns = ['PC1', 'PC2', 'PC3', 'CAREER'])
df_pca_analysis4_cv_y = pd.DataFrame(df_pca_analysis4, columns = ['SALARY'])
X = df_pca_analysis4_cv_x.ix[:]
X


# In[32]:

from sklearn.cross_validation import KFold
cv = KFold(len(df_pca_analysis4_cv_y['SALARY']), n_folds=3, random_state=0)
for train_index, test_index in cv:
    print("test  y:", df_pca_analysis3_cv_y['SALARY'][test_index])
    print("." * 80 )        
    print("train y:", df_pca_analysis3_cv_y['SALARY'][train_index])
    print("=" * 80 )


# In[ ]:

from sklearn.cross_validation import StratifiedKFold
cv = StratifiedKFold(df_pca_analysis4_cv_y['SALARY'], n_folds=4, random_state=42)
for train_index, test_index in cv:
    print("test X:\n", df_pca_analysis4_cv_x.ix[test_index])
    print("." * 80 )        
    print("test y:\n", df_pca_analysis4_cv_y['SALARY'][test_index])
    print("=" * 80 )


# In[ ]:

from sklearn.cross_validation import LabelKFold
cv = LabelKFold(df_pca_analysis4_cv_y['SALARY'], n_folds=3)
for train_index, test_index in cv:
    print("test  y:\n", df_pca_analysis4_cv_y['SALARY'][test_index])
    print("." * 80 )        
    print("train y:\n", df_pca_analysis4_cv_y['SALARY'][train_index])
    print("=" * 80 )


# In[ ]:

print (df_pca_analysis4_cv_x)
print (df_pca_analysis4_cv_y)


# In[ ]:

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

model = LinearRegression()

scores = np.zeros(3)

for i, (train_index, test_index) in enumerate(cv):
    X_train = df_pca_analysis4_cv_x.ix[train_index]
    y_train = df_pca_analysis4_cv_y['SALARY'][train_index]
    X_test = df_pca_analysis4_cv_x.ix[test_index]
    y_test = df_pca_analysis4_cv_y['SALARY'][test_index]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    #scores[i] = mean_squared_error(y_test, y_pred)
    scores[i] = r2_score(y_test, y_pred)

#np.mean(scores)
np.mean(scores)


# In[ ]:

plt.scatter(y_test, y_pred)
plt.show()


# In[ ]:

from sklearn.cross_validation import cross_val_score
cross_val_score(model, df_pca_analysis4_cv_x.ix, df_pca_analysis4_cv_y['SALARY'], "r2_score", cv)


# In[ ]:

df_norm = pd.read_csv('sqrt_norm-2.csv')


df_norm_career = pd.DataFrame(df_norm['N_CAREER'], columns = ['N_CAREER'])

#df_norm_career

del df_norm['N_CAREER']

df_norm.columns = f

df_norm

df2 = pd.read_csv('salary_prediction_taja_add_career_final_test.csv')
dfX2 = pd.DataFrame(df2, columns = d)
dfY2 = pd.DataFrame(df2, columns = ['SALARY'])

dfX2['HEUITA'] += dfX2['HEUIBI']
del dfX2['HEUIBI']

for i in f:
    dfX2[i] = dfX2[i] / df_norm[i]

dfX3 = dfX2.dot(kp)



dfX_CAREER = pd.DataFrame(df2, columns = ['CAREER'])
dfX_CAREER

dfX_CAREER = dfX_CAREER['CAREER'] / df_norm_career['N_CAREER']
dfX_CAREER2 = pd.DataFrame(dfX_CAREER, columns = ['CAREER'])
dfX_CAREER2

df_final_data_set = pd.concat([dfX3, dfX_CAREER2], axis = 1)
df_final_data_set


# In[ ]:

result = np.ones(20)

for i in range(20):
    result[i] = (9.3014 + (-3.6845) * df_final_data_set['PC1'][i] + (1.6428) * df_final_data_set['PC2'][i] + (19.3214) * (df_final_data_set['PC3'][i] ** 2) + 6.5626 * df_final_data_set['CAREER'][i])

result2 = np.ones(20)

for i in range(20):
    result2[i] = np.exp(result[i])/1000
    
    
    
result3 = result2.reshape(-1,1)
    

df_results_temp = pd.DataFrame(result3, columns = ['y_pred'])

df_results = pd.concat([df_results_temp, dfY2], axis = 1)

df_results


