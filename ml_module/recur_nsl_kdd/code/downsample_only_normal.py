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
        "3_R2L": 0.02, 
        "4_U2R": 0.5,
    }

    prep_path = filepaths.PREPROCESS_PATH
    model_path = filepaths.FRAME_MODEL_PATH

    normal_df = pd.read_csv(os.path.join(prep_path, '0_Normal.csv'))
    labels = {
        '2_Probe': ['0_Normal'], 
        '3_R2L': ['0_Normal', '2_Probe'], 
        '4_U2L': ['0_Normal', '2_Probe']
    }

    print("========= Downsample =========")
    drop_indexes = []
    label2centroids = {}
    for label in labels:
        print("--------- {} ---------".format(label))
        # cent_fpath = os.path.join(model_path, "kmeans", label[2:] + "_kmeans_centroids.pkl")
        # with open(cent_fpath, "rb") as f:
        #     centroids = pickle.load(f)
        fpath = os.path.join(filepaths.PREPROCESS_PATH, '{}.csv'.format(label))
        sub_df = pd.read_csv(fpath)

        print(" -- Training Kmeans")
        sub_X = sub_df.drop("attack_type", axis=1).values
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
            
            for n, s in sorted(silh_coefs.items(), key=lambda x: x[1], reverse=True):
                print(n, s / N_TEST)
        
        # print(" -- Calculating calinski_harabasz_score", len(normal_train_X))
        # ch_score = calinski_harabasz_score(sub_X, kmeans.labels_)
        # print("Kmeans ch_score", ch_score)

        print("--- downsample...")
        centroids = kmeans.cluster_centers_
        to_drop = get_bad_majority_set(normal_df, centroids, a=a_arr[label])
        to_remain = filter(lambda x: x not in to_drop, range(len(normal_df)))
        downsampled_sub_df = normal_df.loc[to_remain, :]
        print("Orig samples: {}, then samples: {}".format(len(normal_df), len(downsampled_sub_df)))
        sub_fpath = os.path.join(filepaths.TRAIN_DOWNSAMPLE_FOLDER_PATH, '0_Normal_downsampled_by_' + label + '.csv')
        downsampled_sub_df.to_csv(sub_fpath, index=False)

        drop_indexes.extend(to_drop)
    
    remained_indexes = filter(lambda x: x not in drop_indexes, range(len(normal_df)))
    downsampled_df = normal_df.loc[remained_indexes, :]
    print("Orig all samples: {}, then all samples: {}".format(len(normal_df), len(downsampled_df)))
    fpath_new = os.path.join(filepaths.TRAIN_DOWNSAMPLE_FOLDER_PATH, '0_Normal_downsampled.csv')
    downsampled_df.to_csv(fpath_new, index=False)


    # pca
    if "--pca" in opts or "-p" in opts:
        print("========= PCA PLOTTING =========")

        train_path = filepaths.TRAIN_PATH
        img_path = filepaths.IMG_PATH
        train_df = pd.read_csv(train_path)
        fig, axes = plt.subplots(8, 2, figsize=(15, 24))

        # 读原正负样本数据
        X = train_df.drop('attack_type', axis=1).values
        c = 1
        y = [c for _ in range(len(X))]

        # 用原正负样本数据训练pca
        pca = PCA(n_components=2)
        X_new = pca.fit_transform(X)
        X_all = X_new.copy()
        y_all = y

        # 绘制欠采样数据
        for fname in os.listdir(filepaths.TRAIN_DOWNSAMPLE_FOLDER_PATH):
            fpath = os.path.join(filepaths.TRAIN_DOWNSAMPLE_FOLDER_PATH, fname)
            downsampled_df = pd.read_csv(fpath)
            if fname == "0_Normal_downsampled.csv":
                plt.subplot(6, 3, 2)
                plt.xlim(-2.5, 3)
                plt.ylim(-2, 3.5)
            else:
                label = fname[24:-4]
                label_num = int(label[0])
                plt.subplot(6, 3, label_num * 3 + 2)
                plt.xlim(-2.5, 3)
                plt.ylim(-2, 3.5)
            downsampled_X = downsampled_df.drop("attack_type", axis=1).values
            X_new = pca.transform(downsampled_X)
            color = [0] * len(X_new)
            try:
                centroids_new = pca.transform(label2centroids[label])
                color.extend([1] * len(centroids_new))
                X_new = np.vstack([X_new, centroids_new])
            except:
                pass
            plt.scatter(X_new[:, 0], X_new[:, 1], c=color, linewidths=0.01, marker='o')

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

            c += 1
            y = [c for _ in range(len(X))]

            # pca降维
            X_new = pca.transform(X)

            # 准备画最终图
            if "_test" not in fname:
                X_all = np.vstack((X_all, X_new))
                y_all.extend(y)

            # 绘图
            test = 2 if "_test" in fname else 0
            position = int(label[0]) * 3 + 1 + test
            plt.subplot(6, 3, position)
            plt.scatter(X_new[:, 0], X_new[:, 1], c=y, linewidths=0.01, marker='o')
            plt.xlim(-2.5, 3)
            plt.ylim(-2, 3.5)
            plt.title(fname)

        plt.subplot(6, 3, 17)
        plt.xlim(-2.5, 3)
        plt.ylim(-2, 3.5)
        plt.scatter(X_all[:, 0], X_all[:, 1], c=y_all, linewidths=0.01, marker='o')
        plt.savefig(os.path.join(img_path, "downsample.png"))
