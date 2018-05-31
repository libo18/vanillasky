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
features_new = [ 'day_gap_after','this_month_user_receive_all_coupon_count', 'this_month_user_receive_same_coupon_count', 'merchant_coupon_transfer_rate', 	 'this_month_user_receive_same_coupon_firstone', 'discount_man', 	 'discount_rate','user_merchant_buy_total', 'days_distance',	 'merchant_mean_distance', 'user_merchant_received', 'user_merchant_coupon_transfer_rate', 'day_gap_before', 'user_merchant_any',  'coupon_rate', 'user_merchant_rate', 'buy_total', 'is_man_jian', 'total_sales', 'label']

print("train01")
train01_data = pd.read_csv('train01.csv')
train01_data_new = train01_data[features_new]
train01_data_new_np = train01_data_new.values
imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
train01_data_new = imp.fit_transform(train01_data_new_np)
train01_data_new_df = pd.DataFrame(train01_data_new, columns=features_new)
train01_data_new_df_1 = train01_data_new_df[train01_data_new_df.label >-1]
train01_data_new_df_1_np = train01_data_new_df_1.values
print(train01_data_new_df_1.info())


print("train02")
train02_data = pd.read_csv('train02.csv')
train02_data_new = train02_data[features_new]
train02_data_new_np = train02_data_new.values
imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
train02_data_new = imp.fit_transform(train02_data_new_np)
train02_data_new_df = pd.DataFrame(train02_data_new, columns=features_new)
train02_data_new_df_2 = train02_data_new_df[train02_data_new_df.label >-1]
train02_data_new_df_2_np = train02_data_new_df_2.values
print(train02_data_new_df_2.info())

train_data_new_np = np.concatenate([train01_data_new_df_1,train02_data_new_df_2],axis = 0)

train_data_new_df = pd.DataFrame(train_data_new_np, columns=features_new)
print(train_data_new_df.info())
train_data_new_df.to_csv("processed_data.csv")