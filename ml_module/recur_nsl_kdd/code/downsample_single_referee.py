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


def get_bad_majority_set(maj_df, min_centroids, a=1):
    """
      Find negative (majority) samples which are highly possible to be classified as positive.
      Calculate the average distance to the input centroids for each major class sample. 
    The shorter the avg_dist is, the more similar to minor class this certain sample is, and 
    therefore the more possible this sample is likely to be downsampled(removed).

    param M: neighbor number considered when finding bad set
    param a: hyperparameter. Drop rate formula: dr = exp(-a * avg_dist).
      The larger 'a' is, the less likely points are removed. 'a' acts like tolerance.
      In other words, the less 'a' is, the stronger downsample is.
    return: index of bad set in maj_df
    """
    maj_arr = maj_df.drop('attack_type', axis=1).values
    avg_dist_arr = np.zeros((maj_arr.shape[0], ))
    min_dist_arr = np.zeros((maj_arr.shape[0], ))

    for i, maj_point in enumerate(maj_arr):
        dist_to_centroids = eu_dist_batch(min_centroids, maj_point)
        avg_dist = dist_to_centroids.sum() / len(min_centroids)
        avg_dist_arr[i] = avg_dist
        min_dist_arr[i] = np.min(dist_to_centroids)

    # normalize
    max_avg_dist, min_avg_dist = np.max(avg_dist_arr), np.min(avg_dist_arr)
    avg_dist_arr = (avg_dist_arr - min_avg_dist) / (max_avg_dist - min_avg_dist)
    max_min_dist, min_min_dist = np.max(min_dist_arr), np.min(min_dist_arr)
    min_dist_arr = (min_dist_arr - min_min_dist) / (max_min_dist - min_min_dist)

    print("Avg avg_dist:", np.average(avg_dist_arr))
    print("Min avg_dist:", np.min(avg_dist_arr))
    print("Max avg_dist:", np.max(avg_dist_arr))

    # calculate drop rate
    drop_rates = np.exp(-a * avg_dist_arr)
    # drop_rates = np.exp(-a * min_dist_arr)

    drop_indexes = []
    for i, (maj_point, drop_rate) in enumerate(zip(maj_arr, drop_rates)):
        p = random.random()
        if p < drop_rate:
            drop_indexes.append(i)
    
    return drop_indexes


if __name__ == "__main__":

    opts, args = getopt.getopt(sys.argv[1:], "pa:", ["pca"])
    opts = dict(opts)
    a = float(opts.get("-a", 1))

    kmeans_cent_nums = {
        "0_Normal": 6, #ch 7 32041    silh 6 0.50429
        "1_DOS": 4, #ch 3 120000+  silh 4 0.6071
        "2_Probe": 9, #ch 11 19681 25 20000+ 200 40000+    silh 7 0.66727
        "3_R2L": 7, # ch 4 4389   silh 7 0.73570
        "4_U2R": 6, # silh 6 0.38476
    }

    a_arr = {
        "0_Normal": None, 
        "1_DOS": None, 
        "2_Probe": 0.8, 
        "3_R2L": 0.4, 
        "4_U2R": 0.5,
    }

    prep_path = filepaths.PREPROCESS_PATH
    model_path = filepaths.FRAME_MODEL_PATH

    labels = {
        '2_Probe': ['0_Normal'], 
        '3_R2L': ['0_Normal', '2_Probe'], 
        '4_U2R': ['0_Normal', '2_Probe']
    } # keys are referee and values are downsamplee

    normal_df = pd.read_csv(os.path.join(prep_path, '0_Normal.csv'))
    print("========= Downsample =========")
    label2dropid = {}
    label2centroids = {}
    for label in labels:
        print("--------- referee: {} ---------".format(label))
        fpath = os.path.join(filepaths.PREPROCESS_PATH, '{}.csv'.format(label))
        referee_df = pd.read_csv(fpath)

        # 对referee集合进行聚类
        print(" -- Training Kmeans")
        sub_X = referee_df.drop("attack_type", axis=1).values
        print(" -- sub_X.shape:", sub_X.shape)
        n_clusters = kmeans_cent_nums[label]
        if isinstance(n_clusters, int):
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(sub_X)
            print(" -- Calculating silh_coef")
            silh_coef = silhouette_score(sub_X, kmeans.labels_)
            print("Kmeans silh_coef", silh_coef)
            label2centroids[label] = kmeans.cluster_centers_
        else:
            silh_coefs = Counter()
            N_TEST = 100
            for _ in range(N_TEST):
                for n in n_clusters:
                    kmeans = KMeans(n_clusters=n)
                    kmeans.fit(sub_X)
                    silh_coef = silhouette_score(sub_X, kmeans.labels_)
                    silh_coefs[n] += silh_coef
            # 选择较好的聚类数目
            for n, s in sorted(silh_coefs.items(), key=lambda x: x[1], reverse=True):
                print(n, s / N_TEST)

        # 对每个downsamplee，根据当前referee欠采样
        for downsamplee in labels[label]:
            print("--- downsamplee: {}...".format(downsamplee))
            downsamplee_df = pd.read_csv(os.path.join(prep_path, downsamplee + '.csv'))
            centroids = kmeans.cluster_centers_ # 计算referee集合的聚类中心
            to_drop = get_bad_majority_set(downsamplee_df, centroids, a=a_arr[label]) # 去除downsamplee集合中离referee集合聚类中心较近的点
            to_remain = filter(lambda x: x not in to_drop, range(len(downsamplee_df)))
            downsampled_sub_df = downsamplee_df.loc[to_remain, :]
            print("Orig samples: {}, then samples: {}".format(len(downsamplee_df), len(downsampled_sub_df)))
            sub_fpath = os.path.join(filepaths.TRAIN_DOWNSAMPLE_FOLDER_PATH, downsamplee + '_downsampled_by_' + label + '.csv')
            downsampled_sub_df.to_csv(sub_fpath, index=False)
            try:
                label2dropid[downsamplee].extend(to_drop)
            except KeyError:
                label2dropid[downsamplee] = to_drop
        
        # 对每个downsamplee，根据其对应的所有referee欠采样
        for downsamplee in label2dropid:
            print("--- downsamplee: {}...".format(downsamplee))
            downsamplee_df = pd.read_csv(os.path.join(prep_path, downsamplee + '.csv'))
            remained_indexes = filter(lambda x: x not in label2dropid[downsamplee], range(len(downsamplee_df)))
            downsampled_df = downsamplee_df.loc[remained_indexes, :]
            print("Orig all samples: {}, then all samples: {}".format(len(downsamplee_df), len(downsampled_df)))
            fpath_new = os.path.join(filepaths.TRAIN_DOWNSAMPLE_FOLDER_PATH, downsamplee + '_downsampled.csv')
            downsampled_df.to_csv(fpath_new, index=False)

    # pca
    if "--pca" in opts or "-p" in opts:
        print("========= PCA PLOTTING =========")
        # downsamplee_num = len(label2dropid)
        # referee_num = len(label2centroids)
        downsamplee_num = 5
        referee_num = 5
        print(" --- downsamplee_num:", downsamplee_num)
        print(" --- referee_num:", referee_num)

        train_path = filepaths.TRAIN_PATH
        img_path = filepaths.IMG_PATH
        train_df = pd.read_csv(train_path)
        fig, axes = plt.subplots(downsamplee_num, referee_num + 2, figsize=(36, 24))

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
                position = (downsamplee_label_num+1) * (referee_num + 2) - 1
                print("downsample: {}".format(downsamplee_label), position)
                plt.subplot(downsamplee_num, referee_num + 2, position)
            else:
                p = fname.index("by")
                downsamplee_label = fname[ : p-13]
                downsamplee_label_num = int(downsamplee_label[0])
                referee_label = fname[p+3 : -4]
                referee_label_num = int(referee_label[0])
                position = downsamplee_label_num * (referee_num + 2) + referee_label_num + 1
                print("downsample: {}, referee: {}".format(downsamplee_label, referee_label), position)
                plt.subplot(downsamplee_num, referee_num + 2, position)
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
            test = referee_num + 1 if "_test" in fname else 0
            position = int(label[0]) * (referee_num + 2) + 1 + test
            print(fname, position)
            plt.subplot(downsamplee_num, referee_num + 2, position)
            plt.scatter(X_new[:, 0], X_new[:, 1], linewidths=0.01, marker='o')
            plt.xlim(-2.5, 3)
            plt.ylim(-2, 3.5)
            plt.title(fname)

        plt.savefig(os.path.join(img_path, "downsample.png"))
