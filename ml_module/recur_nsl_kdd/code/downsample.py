import utils.filepaths as filepaths
import os, sys, random, pickle, getopt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score


def divide_raw_train_set(df):
    for attack_type, sub_df in df.groupby('attack_type'):
        sub_df.to_csv(os.path.join(filepaths.TRAIN_SMOTE_FOLDER_PATH, '{}.csv'.format(attack_type)), index=False)


def divide_raw_test_set(df):
    for attack_type, sub_df in df.groupby('attack_type'):
        sub_df.to_csv(os.path.join(filepaths.TRAIN_SMOTE_FOLDER_PATH, '{}_test.csv'.format(attack_type)), index=False)


def eu_dist(x, y):
    return ((y - x) ** 2).sum() ** 0.5


def eu_dist_batch(x, y):
    return ((y - x) ** 2).sum(axis=1) ** 0.5


def random_downsample():
    pass


def get_sample_set(samplee_df, referee_centroids=None, a=1, ramdom_ratio=None):
    """
      Find negative (majority) samples which are highly possible to be classified as positive.
      Calculate the average distance to the input centroids for each major class sample. 
    The shorter the avg_dist is, the more similar to minor class this certain sample is, and 
    therefore the more possible this sample is likely to be downsampled(removed).

    param M: neighbor number considered when finding bad set
    param a: hyperparameter. Drop rate formula: dr = exp(-a * avg_dist).
      The larger 'a' is, the less likely points are removed. 'a' acts like tolerance.
      In other words, the less 'a' is, the stronger sampling is.
    return: indexes of kept points in samplee_df
    """
    
    samplee_arr = samplee_df.drop('attack_type', axis=1).values
    samplee_len = samplee_arr.shape[0]

    if not ramdom_ratio:
        avg_dist_arr = np.zeros((samplee_len, ))
        min_dist_arr = np.zeros((samplee_len, ))

        for i, samplee_point in enumerate(samplee_arr):
            dist_to_centroids = eu_dist_batch(referee_centroids, samplee_point)
            avg_dist = dist_to_centroids.sum() / len(referee_centroids)
            avg_dist_arr[i] = avg_dist
            min_dist_arr[i] = np.min(dist_to_centroids)

        # normalize
        max_avg_dist, min_avg_dist = np.max(avg_dist_arr), np.min(avg_dist_arr)
        avg_dist_arr = (avg_dist_arr - min_avg_dist) / (max_avg_dist - min_avg_dist)
        # max_min_dist, min_min_dist = np.max(min_dist_arr), np.min(min_dist_arr)
        # min_dist_arr = (min_dist_arr - min_min_dist) / (max_min_dist - min_min_dist)

        # calculate drop/keep rate
        rates = np.exp(-a * avg_dist_arr)
        # rates = np.exp(-a * min_dist_arr)

        rtr_indexes = []
        for i, rate in enumerate(rates):
            p = random.random()
            if p < rate:
                rtr_indexes.append(i)

    else:
        # random sample
        print("")
        remain_len = int(ramdom_ratio * samplee_len)
        rtr_indexes = random.sample(range(samplee_len), remain_len)
    
    return rtr_indexes


if __name__ == "__main__":

    opts, args = getopt.getopt(sys.argv[1:], "pa:r:", ["pca", "random="])
    opts = dict(opts)

    RANDOM_RATIO = opts.get("--random") or opts.get("-r")
    if RANDOM_RATIO:
        RANDOM_RATIO = float(RANDOM_RATIO)
    print( "RANDOM:" , RANDOM_RATIO)

    kmeans_cent_nums = {
        "0_Normal": 6, #ch 7 32041    silh 6 0.50429
        "1_DOS": 4, #ch 3 120000+  silh 4 0.6071
        "2_Probe": 8, #ch 11 19681 25 20000+ 200 40000+    silh 8 0.55993
        "3_R2L": 7, # ch 4 4389   silh 7 0.73866
        "4_U2R": 5, # silh 5 0.37584
    }

    prep_path = filepaths.PREPROCESS_PATH
    model_path = filepaths.FRAME_MODEL_PATH
    downsample_path = filepaths.TRAIN_DOWNSAMPLE_FOLDER_PATH

    # downsamplee2referee = {
    #     # '0_Normal' : [('3_R2L', 'drop', 0.4), ('4_U2R', 'drop', 0.5)], # 提升bin acc dr f1, 降低bin prec far 误判1_DOS太多
    #     # '0_Normal' : [('2_Probe', 'drop', 0.8)], 
    #     '0_Normal' : [('1_DOS', 'drop', 8), ('2_Probe', 'drop', 1), ('3_R2L', 'drop', 0.5), ('4_U2R', 'drop', 0.5)], 
    #     '1_DOS' : [('2_Probe', 'drop', 0.8), ('3_R2L', 'drop', 0.4), ('4_U2R', 'drop', 0.5)], # downsample_single_referee.py
    #     # '1_DOS' : [('3_R2L', 'drop', 0.4), ('4_U2R', 'drop', 0.5)], # downsample_single_referee.py
    #     '2_Probe' : [('3_R2L', 'drop', 0.4), ('4_U2R', 'drop', 0.5)], # downsample_single_referee.py
    #     # '2_Probe' : [('0_Normal', 'keep', 0.05), ('1_DOS', 'drop', 0.6)], 
    #     # '2_Probe' : [('0_Normal', 'keep', 0.05), ('1_DOS', 'keep', 0.5), ('3_R2L', 'drop', 0.4), ('4_U2R', 'drop', 0.4)], 
    # } # keys are downsamplee and values are referee

    # downsamplee2referee = { # downsample_single_referee.py
    #     '0_Normal' : [('2_Probe', 'drop', 0.8), ('3_R2L', 'drop', 0.4), ('4_U2R', 'drop', 0.5)], 
    #     '2_Probe' : [('3_R2L', 'drop', 0.4), ('4_U2R', 'drop', 0.5)],
    # } # keys are downsamplee and values are referee

    # downsamplee2referee = { # f1 high
    #     '0_Normal' : [('2_Probe', 'drop', 0.8), ('3_R2L', 'drop', 0.5), ('4_U2R', 'drop', 0.8)], 
    # } # keys are downsamplee and values are referee

    # downsamplee2referee = { # good 4-17 1
    #     '0_Normal' : [('1_DOS', 'drop', 5), ('2_Probe', 'drop', 0.8), ('3_R2L', 'drop', 0.5), ('4_U2R', 'drop', 0.8)], 
    #     '1_DOS' : [('0_Normal', 'drop', 0.8)], 
    #     '2_Probe' : [('0_Normal', 'drop', 1.5), ('1_DOS', 'drop', 1)],
    # } # keys are downsamplee and values are referee

    # downsamplee2referee = { # good 4-17 2
    #     '0_Normal' : [('1_DOS', 'drop', 1), ('2_Probe', 'drop', 1), ('3_R2L', 'drop', 1), ('4_U2R', 'drop', 1)], 
    #     '1_DOS' : [('0_Normal', 'drop', 0.8)], 
    #     '2_Probe' : [('0_Normal', 'drop', 1.5), ('1_DOS', 'drop', 1)],
    # } # keys are downsamplee and values are referee

    downsamplee2referee = { # good 4-17 2
        '0_Normal' : [('1_DOS', 'drop', 1), ('2_Probe', 'drop', 1), ('3_R2L', 'drop', 1), ('4_U2R', 'drop', 1)], 
        '1_DOS' : [('0_Normal', 'drop', 0.8)], 
        '2_Probe' : [('0_Normal', 'drop', 1.5), ('1_DOS', 'drop', 1)],
    } # keys are downsamplee and values are referee

    for fname in os.listdir(downsample_path):
        fpath = os.path.join(downsample_path, fname)
        if os.path.isfile(fpath):
            os.remove(fpath)
             
    normal_df = pd.read_csv(os.path.join(prep_path, '0_Normal.csv'))
    print("========= Downsample =========")
    label2centroids = {}
    for downsamplee in downsamplee2referee:
        print("\n--------- downsamplee: {} ---------".format(downsamplee))
        fpath = os.path.join(filepaths.PREPROCESS_PATH, '{}.csv'.format(downsamplee))
        downsamplee_df = pd.read_csv(fpath)
        to_drop = []
        to_remain = []

        # 根据每个referee，对downsamplee欠采样
        for referee, mode, a in downsamplee2referee[downsamplee]:
            print(" --------- referee: {} ---------".format(referee))
            try:
                centroids = label2centroids[referee]
            except KeyError: # 对referee集合进行聚类
                print(" -- Training Kmeans")
                fpath = os.path.join(filepaths.PREPROCESS_PATH, '{}.csv'.format(referee))
                referee_df = pd.read_csv(fpath)
                sub_X = referee_df.drop("attack_type", axis=1).values
                print(" -- sub_X.shape:", sub_X.shape)
                n_clusters = kmeans_cent_nums[referee]
                kmeans = KMeans(n_clusters=n_clusters)
                kmeans.fit(sub_X)
                print(" -- Calculating silh_coef")
                silh_coef = silhouette_score(sub_X, kmeans.labels_)
                print("    Kmeans silh_coef", silh_coef)
                centroids = kmeans.cluster_centers_ # 计算referee集合的聚类中心
                label2centroids[referee] = centroids
            sampled = get_sample_set(downsamplee_df, centroids, a=a, ramdom_ratio=RANDOM_RATIO) # 去除downsamplee集合中离referee集合聚类中心较近的点
            if mode == "drop":
                to_drop += sampled
                unsampled = set(range(len(downsamplee_df))) - set(sampled)
                downsampled_sub_df = downsamplee_df.loc[unsampled, :]
            elif mode == "keep":
                to_remain += sampled
                downsampled_sub_df = downsamplee_df.loc[sampled, :]
            print(" a: {}, Orig samples: {}, then samples: {}".format(a, len(downsamplee_df), len(downsampled_sub_df)))
            sub_fpath = os.path.join(filepaths.TRAIN_DOWNSAMPLE_FOLDER_PATH, downsamplee + '_downsampled_by_' + referee + '.csv')
            downsampled_sub_df.to_csv(sub_fpath, index=False)

        to_drop, to_remain = set(to_drop), set(to_remain)
        to_drop = to_drop - to_remain
        to_remain = set(range(len(downsamplee_df))) - to_drop
        downsampled_df = downsamplee_df.loc[to_remain, :]
        print(" Orig all samples: {}, then all samples: {}".format(len(downsamplee_df), len(downsampled_df)))
        fpath_new = os.path.join(filepaths.TRAIN_DOWNSAMPLE_FOLDER_PATH, downsamplee + '_downsampled.csv')
        downsampled_df.to_csv(fpath_new, index=False)

    # pca
    if "--pca" in opts or "-p" in opts:
        print("========= PCA PLOTTING =========")
        downsamplee_num = 5
        referee_num = 5
        print(" --- downsamplee_num:", downsamplee_num)
        print(" --- referee_num:", referee_num)

        train_path = filepaths.TRAIN_PATH
        img_path = filepaths.IMG_PATH
        train_df = pd.read_csv(train_path)
        fig, axes = plt.subplots(downsamplee_num, referee_num + 3, figsize=(36, 24))

        # 用原正负样本数据训练pca
        pca = PCA(n_components=2)
        X = train_df.drop('attack_type', axis=1).values
        X_new = pca.fit_transform(X)

        # 绘制欠采样数据
        for fname in os.listdir(filepaths.TRAIN_DOWNSAMPLE_FOLDER_PATH):
            fpath = os.path.join(filepaths.TRAIN_DOWNSAMPLE_FOLDER_PATH, fname)
            downsampled_df = pd.read_csv(fpath)
            if fname.endswith("_downsampled.csv"):
                downsamplee_label = fname[:-16]
                downsamplee_label_num = int(downsamplee_label[0])
                position = (downsamplee_label_num + 1) * (referee_num + 3) - 1
                # print("downsample: {}".format(downsamplee_label), position)
                plt.subplot(downsamplee_num, referee_num + 3, position)
            else:
                p = fname.index("by")
                downsamplee_label = fname[ : p-13]
                downsamplee_label_num = int(downsamplee_label[0])
                referee_label = fname[p+3 : -4]
                referee_label_num = int(referee_label[0])
                position = downsamplee_label_num * (referee_num + 3) + referee_label_num + 2
                # print("downsample: {}, referee: {}".format(downsamplee_label, referee_label), position)
                plt.subplot(downsamplee_num, referee_num + 3, position)
            plt.xlim(-2.5, 3)
            plt.ylim(-2, 3.5)
            plt.title(fname)
            downsampled_X = downsampled_df.drop("attack_type", axis=1).values
            X_new = pca.transform(downsampled_X)
            color = [0] * len(X_new)
            try:
                centroids_new = pca.transform(label2centroids[referee_label])
                color.extend([1] * len(centroids_new))
                X_new = np.vstack([X_new, centroids_new])
            except:
                pass
            plt.scatter(X_new[:, 0], X_new[:, 1], linewidths=0.01, marker='o')

        # 绘制原数据
        labels = ['0_Normal', '1_DOS', '2_Probe', '3_R2L', '4_U2R']
        for fname in os.listdir(prep_path):
            for label in labels:
                if fname.startswith(label):
                    break
            else:
                continue
            fpath = os.path.join(prep_path, fname)
            df = pd.read_csv(fpath)
            X = df.drop('attack_type', axis=1).values

            # pca降维
            X_new = pca.transform(X)

            # 绘图
            test = referee_num + 2 if "_test" in fname else 0
            position = int(label[0]) * (referee_num + 3) + 1 + test
            # print(fname, position)
            plt.subplot(downsamplee_num, referee_num + 3, position)
            plt.scatter(X_new[:, 0], X_new[:, 1], linewidths=0.01, marker='o')
            plt.xlim(-2.5, 3)
            plt.ylim(-2, 3.5)
            plt.title(fname)

        plt.savefig(os.path.join(img_path, "downsample.png"))
