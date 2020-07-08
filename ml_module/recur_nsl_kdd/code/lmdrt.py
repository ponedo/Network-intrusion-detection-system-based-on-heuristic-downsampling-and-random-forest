"""
    将数据用lmdrt算法处理，保存在csv文件中
"""

import random, math
import numpy as np
import pandas as pd
import utils.filepaths as filepaths
from sklearn.neighbors.kde import KernelDensity


def split_dataset(df):
    """
        将数据集随机划分为等大的两部分
    """
    original_size = len(df)
    df1_size = math.ceil(original_size / 2)
    df_index = list(range(original_size))
    random.shuffle(df_index)
    df1_index, df2_index = df_index[:df1_size], df_index[df1_size:]
    df1, df2 = df.loc[df1_index, :], df.loc[df2_index, :]
    return df1, df2


def kde_single_arr(x_series, bandwidth=1.0):
    """
        x_series: 一个pd.Series，即一列数据
        对x_series里面的数据进行核密度估计
    """
    kde = KernelDensity(bandwidth, kernel='gaussian')
    kde.fit(x_series.values.reshape(-1, 1))
    return kde


def kde_all(df):
    """
        返回各维在标签为0或1下的对数条件概率密度
        标签为0和1时的对数条件概率分别记为g和f
    """
    g_arr, f_arr = [], []
    cols = df.columns
    groups = df.groupby('attack_type')
    for att, grp_df in groups:
        for i, col in enumerate(cols):
            print("KDE fitting, col:", i, col)
            if col == 'attack_type':
                continue
            kde = kde_single_arr(grp_df[col])
            if att == 'Normal':
                g_arr.append(kde.score_samples)
            elif att == 'Attack':
                f_arr.append(kde.score_samples)
            else:
                print("Error attack type!")
    return g_arr, f_arr


def kde_transform(df, kde):
    """
        df:     输入数据
        kde:    (g_arr, f_arr)元组
    """
    g_arr, f_arr = kde
    cols = df.columns
    for i, col in enumerate(cols):
        print("KDE transforming, col:", i, col)
        if col == 'attack_type':
            continue
        col_arr = df[col].values.reshape(-1, 1)
        f, g = f_arr[i], g_arr[i]
        print("    hello0")
        tmp = f(col_arr) - g(col_arr)
        print("    hello1")
        df[col] = tmp
        print("    hello2")
    return df


if __name__ == "__main__":
    train_df = pd.read_csv(filepaths.TRAIN_BINARY_PATH)
    test_df = pd.read_csv(filepaths.TEST_BINARY_PATH)

    train_df1, train_df2 = split_dataset(train_df)

    g_arr, f_arr = kde_all(train_df1)

    train_df2 = kde_transform(train_df2, (g_arr, f_arr))
    test_df = kde_transform(test_df, (g_arr, f_arr))

    train_df2.to_csv(filepaths.TRAIN_LMDRT_BINARY_PATH, index=False)
    test_df.to_csv(filepaths.TEST_LMDRT_BINARY_PATH, index=False)
