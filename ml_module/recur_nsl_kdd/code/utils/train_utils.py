"""
    用函数封装了各个训练流程，包括：
    data_label_split:   数据与表签的分离
    oversample:         过采样
    downsample:         欠采样
    get_selected_cols:  特征约简后用来得到剩余特征名称的函数
    GridSearchTestSet:  改变grid_search评估分类器好坏的标准（不是用cv的分数，而是用在test集上的分数）
    train_model:        训练模型，可以使用grid_search
    rfe:                循环特征约简，每一迭代训练分类器，去除分类器所认为权重最低的特征
    cross_validation:   交叉验证
"""


import random
import pandas as pd
import numpy as np
from pprint import pprint
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score
from .metric import confusion_matrix


def data_label_split(df):
    return df.drop('attack_type', axis=1), df.attack_type


def oversample(df):
    # 过采样平衡数据集
    attack_col = df.attack_type
    stat = {}
    for i, att in enumerate(attack_col):
        if att not in stat:
            stat[att] = []
        stat[att].append(i)
    max_class_size = max([len(v) for v in stat.values()])
    for att, index_arr in stat.items():
        more_sample = []
        for _ in range(max_class_size - len(index_arr)):
            more_sample.append(random.choice(index_arr))
        stat[att] = index_arr + more_sample
    final_index_arr = sum(stat.values(), [])
    random.shuffle(final_index_arr)
    print("=== Oversample")
    print("from {} to {} samples".format(len(df), len(final_index_arr)))
    df = df.loc[final_index_arr, :]
    return df


def downsample(df, balance=False, percentile=50):
    attack_col = df.attack_type
    percentile /= 100
    stat = {}
    for i, att in enumerate(attack_col):
        if att not in stat:
            stat[att] = []
        stat[att].append(i)
    if balance:
        for att, index_arr in stat.items():
            downsample_num = round(percentile * len(index_arr))
            stat[att] = random.sample(index_arr, downsample_num)
        final_index_arr = sum(stat.values(), [])
        random.shuffle(final_index_arr)
    else:
        # class weights are same
        class_num = len(stat)
        class_downsample_sum = round(percentile * len(df) / class_num)
        for att, index_arr in stat.items():
            new_sample_index = []
            for _ in range(class_downsample_sum):
                new_sample_index.append(random.choice(index_arr))
            stat[att] = new_sample_index
        final_index_arr = sum(stat.values(), [])
        random.shuffle(final_index_arr)
    print("=== Downsample")
    print("from {} to {} samples".format(len(df), len(final_index_arr)))
    df = df.loc[final_index_arr, :]
    return df


def univar_feat_select(train_X, test_X, train_y, method=f_classif, percentile=50):
    univar_fea_selector = SelectPercentile(method, percentile)
    univar_fea_selector.fit(train_X, train_y)
    old_columns = train_X.columns
    train_X = univar_fea_selector.fit_transform(train_X, train_y)
    new_col_num = train_X.shape[1]
    new_columns = get_selected_cols(old_columns, univar_fea_selector.scores_, new_col_num)
    train_X = pd.DataFrame(train_X, columns=new_columns)
    test_X = pd.DataFrame(univar_fea_selector.transform(test_X), columns=new_columns)
    print("=== Used ANOVA for feature selecting, {} features selected:".format(new_col_num))
    # pprint(new_columns)
    return train_X, test_X, univar_fea_selector


def get_selected_cols(cols, scores, k):
    assert len(cols) == len(scores)
    assert len(cols) >= k
    max_score_cols = list(zip(range(k), scores[:k]))
    max_score_cols.sort(key=lambda x: x[1])
    for i in range(k, len(cols)):
        tmp = scores[i]
        if tmp >= max_score_cols[0][0]:
            max_score_cols[0] = (i, tmp)
            max_score_cols.sort(key=lambda x: x[1])
    return cols[[index for index, _ in max_score_cols]]


class GridSearchTestSet:
    """
    注意！！！
    这个类不应该存在，用test集的结果来指导grid search是错误的
    只是为了尝试复现论文
    """
    def __init__(self, estimator, param_grid, scoring, test_X, test_y):
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.test_X = test_X
        self.test_y = test_y
        self.best_params_ = None
        self.best_score_ = None

    def fit(self, train_X, train_y):
        def make_gen(param_grid):
            # generate a combination of parameters at each yield
            param_grid_list = list(param_grid.items())
            def select_para_n(n, parameters):
                # detemine the n_th parameter
                if n == len(param_grid_list):
                    yield parameters
                else:
                    param_name_n, param_list_n = param_grid_list[n]
                    for val in param_list_n:
                        parameters[param_name_n] = val
                        yield from select_para_n(n + 1, parameters)
            yield from select_para_n(0, {})

        best_score, best_params = float("-inf"), None
        for params in make_gen(self.param_grid):
            clf = self.estimator(**params)
            clf.fit(train_X, train_y)
            score = self.scoring(clf, self.test_X, self.test_y)
            # print(score, best_score)
            if score > best_score:
                best_score = score
                best_params = params
        self.best_params_ = best_params
        self.best_score_ = best_score


def train_model(estimator, train_X, train_y, test_X=None, test_y=None, params={}, grid_search=False, param_grid=None, scoring=None):
    if isinstance(train_X, pd.DataFrame):
        train_X = train_X.values
        train_y = train_y.values

    # Grid Search
    if grid_search:
        # grid_searcher = GridSearchCV(estimator(), param_grid, scoring, cv=10)
        grid_searcher = GridSearchTestSet(estimator, param_grid, scoring, test_X, test_y)
        grid_searcher.fit(train_X, train_y)
        best_param = grid_searcher.best_params_
        print("=== Used Grid Search for param tuning:")
        print('best_params', grid_searcher.best_params_)
        print('best score:', grid_searcher.best_score_)

    # Train classifier
    if grid_search:
        clf = estimator(**best_param)
    else:
        clf = estimator(**params)
    clf.fit(train_X, train_y)
    return clf


def rfe(estimator, train_X, train_y, test_X=None, test_y=None, min_feature_num=15, params={}, grid_search=False, param_grid=None, scoring=None):
    print("+++ RFE STARTS")
    discarded_features = []
    iter_num, feature_num = 0, len(train_X.columns)
    while feature_num > min_feature_num:
        clf = train_model(estimator, train_X, train_y, test_X, test_y, params, grid_search, param_grid, scoring)
        min_feature, min_feature_score = None, float('inf')
        for col, feat_score in zip(train_X.columns, clf.feature_importances_):
            if feat_score < min_feature_score:
                min_feature_score = feat_score
                min_feature = col
        train_X.drop(min_feature, axis=1, inplace=True)
        discarded_features.append(min_feature)
        feature_num -= 1
        iter_num += 1
        print("++ rfe iter {}: discarded feature is {}".format(iter_num, min_feature))
        if test_X is not None and test_y is not None:
            test_y_pred = clf.predict(test_X)
            confusion_mat, score = confusion_matrix(test_y_pred, test_y)
            acc_score = accuracy_score(test_y, test_y_pred)
            print("feature num:", feature_num + 1)
            print("test acc:", acc_score)
            print("test score:", score)
            print("test confusion matrix:", "\n", confusion_mat, "\n")
            test_X.drop(min_feature, axis=1, inplace=True)
    discarded_features.reverse()
    return train_X


def cross_validation(X, y, estimator, params={}, cv=10):
    acc_scores, scores = [], []
    for i in range(cv):
        valid_begin, valid_end = round(i / cv * len(X)), round((i + 1) / cv * len(X))
        if isinstance(X, pd.DataFrame):
            train_X, valid_X = pd.concat([X.loc[:valid_begin-1, :], X.loc[valid_end:, :]]).values, X.loc[valid_begin:valid_end-1, :].values
            train_y, valid_y = pd.concat([y[:valid_begin], y[valid_end:]]).values, y[valid_begin:valid_end].values
        elif isinstance(X, np.ndarray):
            train_X, valid_X = np.vstack([X[:valid_begin, :], X[valid_end:, :]]), X[valid_begin:valid_end, :]
            train_y, valid_y = np.concatenate([y[:valid_begin], y[valid_end:]]), y[valid_begin:valid_end]
        clf = estimator(**params)
        clf.fit(train_X, train_y)
        valid_y_pred = clf.predict(valid_X)
        acc_score = accuracy_score(valid_y, valid_y_pred)
        # confusion_mat, score = confusion_matrix(valid_y_pred, valid_y)
        # print("confusion mat:", "\n", confusion_mat)
        acc_scores.append(acc_score)
        # scores.append(score)
    print("=== {}-fold cross validation:".format(cv))
    print("Avg acc:", sum(acc_scores) / len(acc_scores))
    # print("Avg score:", sum(scores) / len(scores))