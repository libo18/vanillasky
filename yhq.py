# -*- coding: utf-8 -*-
# 加载相关模块和库
import sys
import io
#改变标准输出的默认编码
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
from sklearn.learning_curve import learning_curve
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
import pandas as pd #数据分析
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier

# 用sklearn的learning_curve得到training_score和cv_score，使用matplotlib画出learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1,
                        train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
    """
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    n_jobs : 并行的的任务数(默认1)
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
    if plot:
        plt.figure()
        plt.title(title)
        plt.ylim([0.5, 1])
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(u"训练样本数")
        plt.ylabel(u"得分")
        plt.gca().invert_yaxis()
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"训练集上得分")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"测试集上得分")

        plt.legend(loc="best")

        plt.draw()
        plt.gca().invert_yaxis()
        plt.show()

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff

def set_missing_ages(df):
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])

    predictedAges = np.int64(predictedAges)
    # 用得到的预测结果填补原缺失数据
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges

    return df, rfr

def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"
    df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"
    return df


from sklearn.model_selection import validation_curve
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt



def ModelComplexity(estimator,X, y):
    """ Calculates the performance of the model as model complexity increases.
        The learning and validation errors rates are then plotted. """

    # Create 10 cross-validation sets for training and testing
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

    # Vary the max_depth parameter from 1 to 10
    max_depth = np.arange(1, 100,10)

    # Calculate the training and testing scores
    train_scores, valid_scores = validation_curve(estimator, X, y, \
                                 param_name="n_estimators", param_range=max_depth, cv=cv, scoring=None)
    print(estimator._get_param_names())
    # train_sizes, train_scores, test_scores = learning_curve(
    #     estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

    # Find the mean and standard deviation for smoothing
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    valid_mean = np.mean(valid_scores, axis=1)
    valid_std = np.std(valid_scores, axis=1)


    # Plot the validation curve
    plt.figure(figsize=(7, 5))
    plt.title('classification performance')
    plt.plot(max_depth, train_mean, 'o-', color='r', label='Training Score')
    plt.plot(max_depth, valid_mean, 'o-', color='g', label='Validation Score')
    plt.fill_between(max_depth, train_mean - train_std, \
                    train_mean + train_std, alpha=0.15, color='r')
    plt.fill_between(max_depth, valid_mean - valid_std, \
                    valid_mean + valid_std, alpha=0.15, color='g')

    # Visual aesthetics
    plt.legend(loc='lower right')
    plt.xlabel('Maximum Depth')
    plt.ylabel('score')
    plt.ylim([0.5, 1.05])
    plt.show()


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score,make_scorer
from sklearn.model_selection import GridSearchCV,KFold

cross_validator = KFold(n_splits=10, shuffle=False, random_state=None)

# def performance_metric(y_true, y_predict):
#     """计算并返回真实值相比于预测值的分数"""
#     score = r2_score(y_true, y_predict, sample_weight=None, multioutput=None)
#
#     return score
def grid_scores_to_df(grid_scores):
    """
    Convert a sklearn.grid_search.GridSearchCV.grid_scores_ attribute to a tidy
    pandas DataFrame where each row is a hyperparameter-fold combinatination.
    """
    rows = list()
    for grid_score in grid_scores:
        # for score in enumerate(grid_score.mean_validation_score):
        row = grid_score.parameters.copy()
        # row['fold'] = fold
        row['mean_score'] = grid_score.mean_validation_score
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


from O2O_Coupon_preprocessing_function import GetPreproccessedData
# train01_data = GetPreproccessedData('../yhq/data2/train01.csv','../yhq/data2/train02.csv')
# (1) 读取数据集
# data_train = pd.read_csv("data\\train.csv")
train01_data = pd.read_csv('processed_data.csv', header=0)
# train02_data = pd.read_csv('../yhq/data2/train02.csv', header=0)
#
# test_data = pd.read_csv('../yhq/data2/test.csv', header=0)
print("原始数据统计：")
print(train01_data.info())

# # (2) 特征工程 - 处理缺失值
# data_train, rfr = set_missing_ages(data_train)
# data_train = set_Cabin_type(data_train)#

# print("对Age和Cabin进行缺失值处理后：")
# print(data_train.info())
# (3) 特特工程 - 类目型的特征离散/因子化
# sklearn中preprocessing.OneHotEncoder(), 本例子选用pandas的get_dummies()

# dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin')
# dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')
# dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')
# dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')
# df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
# df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
# print("对Cabin、Embarked、Sex、Pclass进行onecode编码后：")
# print(df.info())

# (4) 特征工程 - 特征抽取
# 我们把需要的feature字段取出来，转成numpy格式，使用scikit-learn中的LogisticRegression建模
# df = data_train_offline
# train_df = df.filter(regex='Survived|Age|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

from sklearn import preprocessing


print("train01")
features = list(train01_data.columns)

imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
train01_data_new = imp.fit_transform(train01_data)
train01_data_new_df = pd.DataFrame(train01_data_new, columns=features)
train01_data_new_df_1 = train01_data_new_df[train01_data_new_df.label >-1]
print(train01_data_new_df_1.info())


# print("train02")
#
# features = list(train02_data.columns)
#
# imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
# train02_data_new = imp.fit_transform(train02_data)
# train02_data_new_df = pd.DataFrame(train02_data_new, columns=features)
# train02_data_new_df_2 = train02_data_new_df[train02_data_new_df.label >-1]
# print(train02_data_new_df_2.info())
#

#train_df.to_csv("processed_titanic.csv" , encoding = "utf-8")
# print("特征选择后：")

train_np = train01_data_new_df_1.as_matrix()

# y即Survival结果
y = train_np[:, -1]
# X即特征属性值
X = train_np[:, :-1]

X = X[:100000,:]
y = y[:100000]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.1,
    random_state=42
)
print("X")
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
# StandardScaler().fit_transform(X)
# Normalizer().fit_transform(X)
#
# #GBDT作为基模型的特征选择
# #print(SelectFromModel(GradientBoostingClassifier()).fit_transform(iris.data, iris.target))
# selector = SelectFromModel(GradientBoostingClassifier(),threshold=None).fit(X, y)
# print("features的重要性排序：")
# print(X[1:10])
# X = selector.transform(X)
# print(X[1:10])

# (5) 模型构建与训练
# clf = linear_model.LogisticRegression(C=100.0, penalty='l1', tol=1e-6)
clf = RandomForestClassifier(criterion='gini', max_depth=5, n_estimators=5)



from sklearn.metrics import classification_report
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))

# # #
# # # print("======================ROC==================================")
# Set the parameters by cross-validation


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
parameter_space = {
    "n_estimators": [10, 15, 20],
    "criterion": ["gini", "entropy"],
    "min_samples_leaf": [2, 4, 6],
}

# scores = ['precision', 'recall', 'roc_auc']
scores = ['roc_auc']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = RandomForestClassifier(random_state=14)
    grid = GridSearchCV(clf, parameter_space, cv=5, scoring='%s' % score)
    # scoring='%s_macro' % score：precision_macro、recall_macro是用于multiclass/multilabel任务的
    grid.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(grid.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    bclf = grid.best_estimator_
    bclf.fit(X_train, y_train)
    y_true = y_test
    y_pred = bclf.predict(X_test)
    y_pred_pro = bclf.predict_proba(X_test)
    y_scores = pd.DataFrame(y_pred_pro, columns=bclf.classes_.tolist())[1].values
    print(classification_report(y_true, y_pred))
    auc_value = roc_auc_score(y_true, y_scores)

    # 绘制ROC曲线
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1.0)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', linewidth=lw, label='ROC curve (area = %0.4f)' % auc_value)
    plt.plot([0, 1], [0, 1], color='navy', linewidth=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
# #





# import xgboost as xgb
# clf = xgb.XGBClassifier(max_depth = 5, n_estimators = 5)

# from sklearn.ensemble import  GradientBoostingClassifier
# clf = GradientBoostingClassifier()
# (6) 绘制learning curve
# plot_learning_curve(clf, u"学习曲线", X, y)
#clf = DecisionTreeClassifier()

# 根据不同的最大深度参数，生成复杂度曲线
# ModelComplexity(clf,X, y)

#
#
# params = {'max_depth':  np.arange(1,10,4),
#           'n_estimators': np.arange(1, 100, 40)}
#
# # scoring_fnc = make_scorer(performance_metric)
#
# # cross_validator = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
# grid = GridSearchCV(estimator=clf, param_grid=params, scoring=None, cv=cross_validator)
#
# # 基于输入数据 [X,y]，进行网格搜索
# grid = grid.fit(X, y)
#
#
# df = grid_scores_to_df(grid.grid_scores_)
# df = df[['max_depth','n_estimators','mean_score']]
# # print(df)
# plt.scatter(df['max_depth'],df['n_estimators'],c=df['mean_score'],cmap=plt.cm.Blues,s=100)
# plt.scatter(grid.best_params_['max_depth'],grid.best_params_['n_estimators'],c=grid.best_score_,cmap=plt.cm.Blues,marker='+',s=200)
# plt.xlabel('max_depth')
# plt.ylabel('n_estimators')
# plt.title('scores(%s)'%clf.__class__.__name__)
# plt.colorbar()
# plt.show()
#
# print("best param" + str(grid.best_params_))
# print("best score" + str(grid.best_score_))