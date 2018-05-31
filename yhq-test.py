
# coding: utf-8

# In[5]:

import numpy as np
import pandas as pd


# In[2]:

# 加载数据
# train_01 = pd.read_csv('E:/02Work/AI/12DAY09-180512-自由讨论/数据集/train01.csv')
train_02 = pd.read_csv('E:/02Work/AI/12DAY09-180512-自由讨论/数据集/train02.csv')


# In[3]:

# train_01.shape


# In[4]:

# 查看数据
train_02.head(10)


# In[ ]:

train_02.shape


# In[6]:

train_02.info()


# In[9]:

# train_02.describe
# 去除user_id 
train_02.drop(['user_id'],axis=1,inplace=True)
# 去除标签中 -1的数据，会给后边处理造成问题，并且-1数量为3336，占比例较小
train_02.label[train_02.label == -1].value_counts()
train_02 = train_02[train_02.label > -1]
train_02.shape


# In[10]:

from sklearn import preprocessing
from sklearn import feature_selection


# In[11]:

# 缺失值计算(也可用pandas.fillna函数)
imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
train_02_new = imp.fit_transform(train_02)
# train_02 = pd.DataFrame(train_02_new)


# In[12]:

train_02_data = train_02_new[:,:-1]
train_02_label = train_02_new[:,-1:]


# In[13]:

# 特征选择，使用GBDT
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier 


# In[14]:

selector = SelectFromModel(GradientBoostingClassifier()).fit(train_02_data, train_02_label)
data = selector.transform(train_02_data)
print(data)
print(selector.estimator_.feature_importances_)


# In[15]:

data.shape


# In[25]:

train_02.columns


# In[34]:

features = {}
for idx,col in enumerate(selector.estimator_.feature_importances_):
    features[train_02.columns[idx]] = col


# In[38]:

print(features)


# In[ ]:




# In[ ]:



