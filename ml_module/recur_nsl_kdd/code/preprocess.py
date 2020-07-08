"""
    预处理
    将数据处理为特定要求所需的形式，存储为csv文件
"""


import os
import pandas as pd
import numpy as np
import utils.datainfo as datainfo
import utils.filepaths as filepaths


def str2num(s):
    try:
        s = float(s) if '.' in s else int(s)
    except ValueError:
        pass
    return s


def read_raw_data(filepath):
    lines = []
    with open(filepath, 'r') as f:
        for line in f:
            features = [str2num(s) for s in line.strip().split(',')]
            # assert len(features) == len(datainfo.feature_list)
            lines.append(features)
    df = pd.DataFrame(lines)
    df.columns = datainfo.feature_list
    return df


def remove_useless_feature(df):
    """
        num_outbound_cmds为全0，应该删除
    """
    for col in df.columns:
        if df[col].nunique() <= 1:
            df = df.drop(col, axis=1)
    return df


def merge_labels(df):
    """
        将label从 正常 和几十小类攻击
        合并为 正常 和 4大类攻击 5种
    """
    att2major = {}
    for major, atts in datainfo.attack_types.items():
        for att in atts:
            att2major[att] = major
    label_col = df['attack_type'][:]
    for row, att in enumerate(label_col):
        label_col[row] = att2major[att]
    df['attack_type'] = label_col
    return df


def merge_labels_binary(df):
    """
        将label从 正常 和4种攻击
        合并为 正常 和 攻击 两种
    """
    label_col = df['attack_type'][:]
    for row, att in enumerate(label_col):
        label_col[row] = '0_Normal' if att == '0_Normal' or att == 'normal' else '1_Attack'
    df['attack_type'] = label_col
    return df


def stat_metadata(df):
    """
    param df: csv filepath or dataframe
    """
    metadata = {}.fromkeys(datainfo.categorical_multi_feature_list)
    if type(df) is str:
        df = pd.read_csv(df)
    
    ## 统计字符特征取值
    for col in datainfo.categorical_multi_feature_list:
        s = set(df[col])
        metadata[col] = s

    ## 统计标签分布
    attack_types_info = {}
    att_col = df.attack_type
    for att in att_col:
        if att not in attack_types_info:
            attack_types_info[att] = 0
        attack_types_info[att] += 1
    print(attack_types_info)

    return metadata


def one_hot(df):
    encode_cols = datainfo.categorical_multi_feature_list
    for col in encode_cols:
        values = datainfo.categorical_multi_feature_metalist[col]
        val2index, i = {}, 0
        for v in values:
            val2index[v] = i
            i += 1
        one_hot_mat = np.zeros((len(df), len(values)))
        for row, val in enumerate(df[col]):
            one_hot_mat[row, val2index[val]] = 1
        one_hot_df = pd.DataFrame(one_hot_mat)
        one_hot_df.columns = [col + "__" + v for v in values]
        df = pd.concat([df, one_hot_df], axis=1)
    df.drop(encode_cols, axis=1, inplace=True)
    return df


def factorize(df):
    encode_cols = datainfo.categorical_multi_feature_list
    for col in encode_cols:
        df[col] = df[col].factorize()[0]
    return df


def find_sparse_feature(df):
    """
        find sparse categorical features
        sparse means the feature identifies only one certain attack type
    """
    label_col = df['attack_type']
    sparse_relations = {}
    to_check_cols = [
        col \
        for col in df.columns \
        if col.split('__')[0] in datainfo.categorical_multi_feature_list
    ]
    for col in to_check_cols:
        sub_df = pd.concat([df[col], label_col], axis=1)
        for val, grp in sub_df.groupby(col):
            if not val == 1:
                continue
            att_types = set(grp['attack_type'])
            if len(att_types) == 1:
                att_type = next(iter(att_types))
                sparse_relations[col] = att_type
    # 反转sparse_relations
    relations_tmp = sparse_relations
    sparse_relations = {}
    for col, att in relations_tmp.items():
        try:
            sparse_relations[att].append(col)
        except KeyError:
            sparse_relations[att] = []
            sparse_relations[att].append(col)
    return sparse_relations


def merge_sparse_feature(df, sparse_relations):
    """
        merge features which identify a certain common attack type into one feature
    """
    for att_type, cols in sparse_relations.items():
        if len(sparse_relations[att_type]) <= 1:
            continue
        new_col = []
        sub_df = df[cols]
        sub_df = sub_df.sum(axis=1) #横向sum
        for row in sub_df:
            new_col.append(0 if row == 0 else 1)
        new_col_name = att_type + '_indicator'
        df[new_col_name] = new_col
        df.drop(cols, axis=1, inplace=True)
    return df


def rearrange(df):
    """
        将标签列和difficulty列放置在dataframe的最末尾
    """
    df_attack_type, _ = df['attack_type'], df['difficulty']
    df.drop(['attack_type', 'difficulty'], axis=1, inplace=True)
    df = pd.concat([df, df_attack_type], axis=1)
    return df


def gaussian_scale(df):
    """
        用公式 (x - μ) / σ
        对数据进行归一化
    """
    numeric_cols = datainfo.numeric_int_feature_list + datainfo.numeric_float_feature_list
    for col in numeric_cols:
        if col == "num_outbound_cmds":
            continue
        mean, std = df[col].mean(), df[col].std()
        df[col] = (df[col] - mean) / std
    return df


def max_min_scale(df, clip_percentile=99, clip_values=None):
    """
        首先截取数据离异点（通过数据值百分比），设定max_x、

        然后用公式 (x - min_x) / (max_x - min_x)
        对数据进行归一化

        显然one-hot出来的维度不需要处理
    """
    if not clip_values is None:
        for col in clip_values:
            series = df[col][:]
            clip_value = clip_values[col]
            min_value = series.min()
            series = (series - min_value) / (clip_value - min_value)
            for i, v in enumerate(series):
                if v > 1:
                    series[i] = 1
            df[col] = series
        return df

    def clip_by_percentile(series, percentile):
        """
            保留percentile比例的有效特征值
        """
        sorted_values = series.sort_values(ascending=False).values
        clip_value_position = int((100 - percentile) * len(sorted_values) / 100)
        clip_value = sorted_values[clip_value_position]
        min_value, max_value = series.min(), series.max()
        assert not min_value == max_value
        if clip_value == min_value:
            # 在这一列有percentile以上的值是一样的，此时找到此列中不同于该多数值的最小值即可
            l = list(sorted_values)
            new_clip_value_index = l.index(clip_value) - 1
            new_clip_value = l[new_clip_value_index]
            clip_value = new_clip_value
        return clip_value

    numeric_cols = datainfo.numeric_int_feature_list + datainfo.numeric_float_feature_list
    clip_values = {}
    for col in numeric_cols:
        if col == "num_outbound_cmds":
            continue
        series = df[col][:]
        clip_value = clip_by_percentile(series, clip_percentile)
        clip_values[col] = clip_value
        min_value = series.min()
        series = (series - min_value) / (clip_value - min_value)
        for i, v in enumerate(series):
            if v > 1:
                series[i] = 1
        df[col] = series
    return df, clip_values


def divide_raw_train_set(df):
    for attack_type, sub_df in df.groupby('attack_type'):
        sub_df.to_csv(os.path.join(filepaths.PREPROCESS_PATH, '{}.csv'.format(attack_type)), index=False)


def divide_raw_test_set(df):
    for attack_type, sub_df in df.groupby('attack_type'):
        sub_df.to_csv(os.path.join(filepaths.PREPROCESS_PATH, '{}_test.csv'.format(attack_type)), index=False)


if __name__ == "__main__":
    train_df = read_raw_data(filepaths.TRAIN_TXT_PATH)
    train_df.to_csv("d:/test.csv")
    # train_df = merge_labels(train_df)
    train_df = merge_labels_binary(train_df)
    train_df = remove_useless_feature(train_df)
    metadata = stat_metadata(train_df)
    train_df = one_hot(train_df)
    sparse_relations = find_sparse_feature(train_df)
    # train_df = merge_sparse_feature(train_df, sparse_relations)
    train_df = rearrange(train_df)
    # train_df = gaussian_scale(train_df)
    train_df, clip_values = max_min_scale(train_df, clip_percentile=99)
    # train_df.to_csv(filepaths.TRAIN_PATH, index=False)
    train_df.to_csv(filepaths.TRAIN_BINARY_PATH, index=False)

    test_df = read_raw_data(filepaths.TEST_TXT_PATH)
    # test_df = merge_labels(test_df)
    test_df = merge_labels_binary(test_df)
    test_df = remove_useless_feature(test_df)
    metadata = stat_metadata(test_df)
    test_df = one_hot(test_df)
    # sparse_relations = find_sparse_feature(test_df)
    # test_df = merge_sparse_feature(test_df, sparse_relations)
    test_df = rearrange(test_df)
    # test_df = gaussian_scale(test_df)
    test_df = max_min_scale(test_df, clip_values=clip_values)
    # test_df.to_csv(filepaths.TEST_PATH, index=False)
    test_df.to_csv(filepaths.TEST_BINARY_PATH, index=False)

    print("========= Dividing raw train dataset =========")
    divide_raw_train_set(train_df)
    test_df = pd.read_csv(filepaths.TEST_PATH)
    print("========= Dividing raw test dataset =========")
    divide_raw_test_set(test_df)