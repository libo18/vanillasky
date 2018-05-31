import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from numpy import vstack, array, nan
from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn import feature_selection
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import SelectFromModel

train_df = pd.read_csv('train01.csv')

imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
train_df_new = imp.fit_transform(train_df)
train_df_feature = train_df_new[:, :-1]
train_df_label = train_df_new[:,-1:]

selector = SelectFromModel(GradientBoostingClassifier()).fit(train_df_feature, train_df_label)
train_df = selector.transform(train_df_feature)
print(train_df)
print(selector.estimator_.feature_importances_)