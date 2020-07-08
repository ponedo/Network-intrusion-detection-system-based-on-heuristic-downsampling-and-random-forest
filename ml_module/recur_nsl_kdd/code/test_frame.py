import os, sys, getopt, pickle
import utils.filepaths as filepaths
import pandas as pd
import numpy as np

from utils.train_utils import *
from utils.metric import *


SMOTE = False
DOWNSAMPLE = False
ANOVA, ANOVA_PERCENTILE = False, 15
KPCA = False
CROSS_VALIDATION = True


def my_predict(test_X):
    global model_name

    # 加载模型Kmeans聚类中心
    model_path = filepaths.FRAME_MODEL_PATH
    kmeans_save_path = os.path.join(model_path, "kmeans")
    centroids = {}
    for fname in os.listdir(kmeans_save_path):
        label = fname[:-21]
        fpath = os.path.join(kmeans_save_path, fname)
        with open(fpath, "rb") as f:
            centroids[label] = pickle.load(f)

    # 加载ANOVA
    if ANOVA:
        univar_fea_select_fpath = os.path.join(model_path, "anova", "univar_fea_selector.pkl")
        with open(univar_fea_select_fpath, "rb") as f:
            univar_fea_selector = pickle.load(f)
        test_X = univar_fea_selector.transform(test_X)
    
    # 加载二分类分类器
    biclf_save_path = os.path.join(model_path, model_name)
    biclfs = {}
    for fname in os.listdir(biclf_save_path):
        label = fname[len(model_name)+1:-4]
        fpath = os.path.join(biclf_save_path, fname)
        with open(fpath, "rb") as f:
            biclfs[label] = pickle.load(f)
    
    pred_Y = []
    for i, test_x in enumerate(test_X):
        if i % 100 == 0:
            print(i)

        test_x = test_x.reshape(1, -1)
        # 计算到每类所有聚类中心的平均距离，判断应该使用哪一个分类器
        min_avg_dist = float("inf")
        prelim_label = None
        for label in centroids:
            avg_dist = eu_dist_batch(centroids[label], test_x).sum() / len(centroids[label])
            if avg_dist < min_avg_dist:
                min_avg_dist = avg_dist
                prelim_label = label

        biclf = biclfs[prelim_label]
        # test_x = univar_fea_selector.transform(test_x)
        pred_y = biclf.predict(test_x)[0]
        pred_Y.append(pred_y)
    
    return np.array(pred_Y)


def my_predict_batch(test_X):
    global model_name

    # 加载模型Kmeans聚类中心
    model_path = filepaths.FRAME_MODEL_PATH
    kmeans_save_path = os.path.join(model_path, "kmeans")
    centroids = {}
    for fname in os.listdir(kmeans_save_path):
        label = fname[:-21]
        fpath = os.path.join(kmeans_save_path, fname)
        with open(fpath, "rb") as f:
            centroids[label] = pickle.load(f)

    # 加载ANOVA
    if ANOVA:
        univar_fea_select_fpath = os.path.join(model_path, "anova", "univar_fea_selector.pkl")
        with open(univar_fea_select_fpath, "rb") as f:
            univar_fea_selector = pickle.load(f)
        test_X = univar_fea_selector.transform(test_X)
    
    # 加载二分类分类器
    biclf_save_path = os.path.join(model_path, model_name)
    biclfs = {}
    for fname in os.listdir(biclf_save_path):
        label = fname[len(model_name)+1:-4]
        fpath = os.path.join(biclf_save_path, fname)
        with open(fpath, "rb") as f:
            biclfs[label] = pickle.load(f)
    
    # 计算到每类所有聚类中心的平均距离，为每个样本分配分类器
    labels = biclfs.keys()
    test_Xs = {}
    test_indexes = {}
    for k in labels:
        test_Xs[k] = []
        test_indexes[k] = []
    for i, test_x in enumerate(test_X):
        min_avg_dist = float("inf")
        prelim_label = None
        for label in centroids:
            avg_dist = eu_dist_batch(centroids[label], test_x).sum() / len(centroids[label])
            if avg_dist < min_avg_dist:
                min_avg_dist = avg_dist
                prelim_label = label
        test_Xs[prelim_label].append(test_x)
        test_indexes[prelim_label].append(i)

    pred_Y = np.array(['0_Normal' for _ in range(len(test_X))])
    for label in labels:
        biclf = biclfs[label]
        test_label_X = np.array(test_Xs[label])
        print(label)
        print(test_label_X.shape)
        pred_y = biclf.predict(test_label_X)
        pred_Y[test_indexes[label]] = pred_y
    
    return pred_Y


if __name__ == "__main__":
    global model_name

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

    # 读测试数据集
    test_df = pd.read_csv(filepaths.TEST_PATH)
    test_X, test_y = data_label_split(test_df)
    test_X, test_y = test_X.values, test_y.values

    # pred_y = my_predict(test_X)
    pred_y = my_predict_batch(test_X)
    acc_score = accuracy_score(test_y, pred_y)
    confusion_mat = confusion_matrix(pred_y, test_y, confusion_mat_only=True)
    bin_metric = binary_metric(pred_y, test_y)
    print("=== Results of {}:".format(model_name))
    print("Final acc:", acc_score)
    print("Final binary metric:", bin_metric)
    print("Final confusion_matrix:", "\n", confusion_mat)