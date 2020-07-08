import os, sys, getopt, pickle
import pandas as pd
import utils.filepaths as filepaths
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.utils import shuffle
from collections import Counter

from utils.train_utils import *
from utils.metric import *


SMOTE = False
ANOVA, ANOVA_PERCENTILE = False, 15
DOWNSAMPLE = False
KPCA = False


def score_func(estimator, X, true_y):
    pred_y = estimator.predict(X)
    return confusion_matrix(pred_y, true_y, confusion_mat_only=True)


if __name__ == "__main__":

    # Parse cmd args
    """
    --model -m [svm|rf|gdbt]
    --smote -s
    --downsample -d
    --anova -a
    --kpca -k
    --grid-search -g
    --cv -v
    --rfe -r
    """
    opts, args = getopt.getopt(
        sys.argv[1:], "m:sdakgvr", 
        ["model=", "smote", "downsample", "anova", "kpca", "grid-search", "cv", "rfe"])
    opts = dict(opts)
    if "-m" in opts:
        model_name = opts["-m"]
    elif "--model" in opts:
        model_name = opts["--model"]
    
    # Read options
    print("=== Options ===")
    SMOTE = SMOTE or "--smote" in opts or "-s" in opts
    print("SMOTE:", SMOTE)
    DOWNSAMPLE = DOWNSAMPLE or "--downsample" in opts or "-d" in opts
    print("DOWNSAMPLE:", DOWNSAMPLE)
    ANOVA = ANOVA or "--anova" in opts or "-a" in opts
    print("ANOVA:", ANOVA)
    KPCA = KPCA or "--kpca" in opts or "-k" in opts
    print("KPCA:", KPCA)

    kmeans_cent_nums = {
        "Normal": 6, #ch 7 32041    silh 6 0.50429
        "DOS": 4, #ch 3 120000+  silh 4 0.6071
        "Probe": 8, #ch 11 19681 25 20000+ 200 40000+    silh 8 0.55993
        "R2L": 7, # ch 4 4389   silh 7 0.73866
        "U2R": 5, # silh 5 0.37584
    }
    
    # 创建存储模型文件夹
    model_path = filepaths.FRAME_MODEL_PATH
    smote_path = filepaths.TRAIN_SMOTE_FOLDER_PATH
    prep_path = filepaths.PREPROCESS_PATH

    # 加载ANOVA
    univar_fea_select_fpath = os.path.join(model_path, "anova", "univar_fea_selector.pkl")
    with open(univar_fea_select_fpath, "rb") as f:
        univar_fea_selector = pickle.load(f)


    #################################
    # 首先对 所有 数据进行Kmeans聚类 #
    #################################

    # 读Normal集
    normal_train_fname = "0_Normal.csv"
    normal_train_fpath = os.path.join(prep_path, normal_train_fname)
    normal_train_df = pd.read_csv(normal_train_fpath)
    
    # 对Normal集Kmeans聚类，Normal集要经过ANOVA特征选择
    normal_train_X = normal_train_df.drop("attack_type", axis=1).values
    if ANOVA:
        normal_train_X = univar_fea_selector.transform(normal_train_X)

    print("=== Training Kmeans: Normal")
    n_clusters = kmeans_cent_nums["Normal"]
    if isinstance(n_clusters, int):
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(normal_train_X)
        print(" -- Calculating silh_coef")
        silh_coef = silhouette_score(normal_train_X, kmeans.labels_)
        print("Kmeans silh_coef", silh_coef)
        centroids = kmeans.cluster_centers_
    else:
        silh_coefs = Counter()
        N_TEST = 1
        for _ in range(N_TEST):
            for n in n_clusters:
                print("  test", n)
                kmeans = KMeans(n_clusters=n)
                kmeans.fit(normal_train_X)
                silh_coef = silhouette_score(normal_train_X, kmeans.labels_)
                silh_coefs[n] += silh_coef
        
        for n, s in sorted(silh_coefs.items(), key=lambda x: x[1], reverse=True):
            print(n, s / N_TEST)

    save_fpath = os.path.join(model_path, "kmeans", "Normal_kmeans_centroids.pkl")
    with open(save_fpath, "wb") as f:
        pickle.dump(centroids, f)

    ###################################
    # 对不同攻击类型数据进行Kmeans聚类 #
    ###################################
    labels = ['1_DOS', '2_Probe', '3_R2L', '4_U2R']
    for label in labels:
        if SMOTE:
            attack_train_fname = label + "_oversampled.csv"
            attack_train_fpath = os.path.join(smote_path, attack_train_fname)
            try:
                attack_train_df = pd.read_csv(attack_train_fpath)
            except FileNotFoundError:
                attack_train_fname = label + ".csv"
                attack_train_fpath = os.path.join(prep_path, attack_train_fname)
                attack_train_df = pd.read_csv(attack_train_fpath)
        else:
            attack_train_fname = label + ".csv"
            attack_train_fpath = os.path.join(prep_path, attack_train_fname)
            attack_train_df = pd.read_csv(attack_train_fpath)
        attack_type = label[2:]

        # Kmeans聚类
        print("=== Training Kmeans:", label)
        attack_train_X = attack_train_df.drop("attack_type", axis=1).values
        print(" -- X size:", len(attack_train_X))
        if ANOVA:
            attack_train_X = univar_fea_selector.transform(attack_train_X)

        n_clusters = kmeans_cent_nums[attack_type]
        if isinstance(n_clusters, int):
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(attack_train_X)
            print(" -- Calculating silh_coef")
            silh_coef = silhouette_score(attack_train_X, kmeans.labels_)
            print("Kmeans silh_coef", silh_coef)
            centroids = kmeans.cluster_centers_
        else:
            silh_coefs = Counter()
            N_TEST = 25
            for _ in range(N_TEST):
                for n in n_clusters:
                    print("  test", N_TEST, n)
                    kmeans = KMeans(n_clusters=n)
                    kmeans.fit(attack_train_X)
                    silh_coef = silhouette_score(attack_train_X, kmeans.labels_)
                    silh_coefs[n] += silh_coef
            
            for n, s in sorted(silh_coefs.items(), key=lambda x: x[1], reverse=True):
                print(n, s / N_TEST)

        save_fpath = os.path.join(model_path, "kmeans", attack_type + "_kmeans_centroids.pkl")
        with open(save_fpath, "wb") as f:
            pickle.dump(centroids, f)
