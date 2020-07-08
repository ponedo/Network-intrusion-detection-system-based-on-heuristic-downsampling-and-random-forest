import os, sys, getopt, pickle
import pandas as pd
import utils.filepaths as filepaths
from nn import DNNModel
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

from utils.train_utils import *
from utils.metric import *


SMOTE = False
DOWNSAMPLE = False
KMEANS = False
ANOVA, ANOVA_PERCENTILE = False, 15
KPCA = False
CROSS_VALIDATION = True


# def score_func(estimator, X, true_y):
#     pred_y = estimator.predict(X)
#     return confusion_matrix(pred_y, true_y, confusion_mat_only=True)


if __name__ == "__main__":

    # Parse cmd args
    """
    --model -m [svm|rf|gdbt]
    --smote -s
    --downsample -d
    --anova -a
    --kpca
    --kmeans
    --grid-search -g
    --cv -v
    --rfe -r
    """
    opts, args = getopt.getopt(
        sys.argv[1:], "m:sdagvr", 
        ["model=", "smote", "downsample", "anova", "kpca", "kmeans", "grid-search", "cv", "rfe"])
    opts = dict(opts)
    if "-m" in opts:
        model_name = opts["-m"]
    elif "--model" in opts:
        model_name = opts["--model"]
    else:
        model_name = GradientBoostingClassifier
    
    # Read options
    print("=== Options ===")
    SMOTE = SMOTE or "--smote" in opts or "-s" in opts
    print("SMOTE:", SMOTE)
    DOWNSAMPLE = DOWNSAMPLE or "--downsample" in opts or "-d" in opts
    print("DOWNSAMPLE:", DOWNSAMPLE)
    ANOVA = ANOVA or "--anova" in opts or "-a" in opts
    print("ANOVA:", ANOVA)
    KPCA = KPCA or "--kpca" in opts
    print("KPCA:", KPCA)
    KMEANS = KMEANS or "--kmeans" in opts
    print("KMEANS:", KMEANS)
    
    # Read input model
    if model_name == "svm":
        model = SVC
        determined_params = {
            'kernel': 'linear',
            'gamma': 'scale'
        }
        param_grid = {
            'kernel': ['rbf'], 
            'C': [0.01, 0.05, 0.1, 0.5, 1, 5], 
            # 'gamma': [0.01]
        }
    elif model_name == "rf":
        model = RandomForestClassifier
        determined_params = {
            'n_estimators': 100, 
            'n_jobs': 2
        }
        param_grid = {
            'criterion': ['gini', 'entropy'], 
            'n_estimators': range(10, 100, 10), 
            # 'max_depth': [None], 
            'min_samples_split': range(2, 10), 
            'min_samples_leaf': range(1, 5), 
            # 'max_leaf_nodes': [None]
        }
    elif model_name == "gbdt":
        model = GradientBoostingClassifier
        determined_params = {
            'n_estimators': 50, 
        }
        param_grid = {
            'criterion': ['gini', 'entropy'], 
            'n_estimators': range(10, 100, 10), 
            # 'max_depth': [None], 
            'min_samples_split': range(2, 10), 
            'min_samples_leaf': range(1, 5), 
            # 'max_leaf_nodes': [None]
        }
    elif model_name == "dnn":
        model = DNNModel
        determined_params = {}
        param_grid = {}

    kmeans_cent_nums = {
        "Normal": 6, #ch 7 32041    silh 6 0.50429
        "DOS": 4, #ch 3 120000+  silh 4 0.6071
        "Probe": 9, #ch 11 19681 25 20000+ 200 40000+    silh 7 0.66727
        "R2L": 7, # ch 4 4389   silh 7 0.73570
        "U2R": 6, # silh 6 0.38476
    }
    
    # 创建存储模型文件夹
    model_path = filepaths.FRAME_MODEL_PATH
    model_save_path = os.path.join(model_path, model_name)
    if model_name not in os.listdir(model_path):
        os.mkdir(model_save_path)
    smote_path = filepaths.TRAIN_SMOTE_FOLDER_PATH
    prep_path = filepaths.PREPROCESS_PATH

    # 读Normal集
    normal_train_fname = "0_Normal.csv"
    normal_test_fname = "0_Normal_test.csv"
    normal_train_fpath = os.path.join(prep_path, normal_train_fname)
    normal_test_fpath = os.path.join(prep_path, normal_test_fname)
    normal_train_df = pd.read_csv(normal_train_fpath)
    normal_test_df = pd.read_csv(normal_test_fpath)


    # ################################
    # # 首先对 所有 数据进行二分类训练 #
    # ################################
    # train_df = pd.read_csv(filepaths.TRAIN_BINARY_PATH)
    # test_df = pd.read_csv(filepaths.TEST_BINARY_PATH)
    # train_X, train_y = data_label_split(train_df)
    # test_X, test_y = data_label_split(test_df)
    # print("=== Normal: normal samples: {}, attack samples: {}".format(train_y.value_counts()['0_Normal'], train_y.value_counts()['1_Attack']))

    # # Univariate feature selection
    # if ANOVA:
    #     train_X, test_X, univar_fea_selector = univar_feat_select(train_X, test_X, train_y, method=f_classif, percentile=ANOVA_PERCENTILE)
    #     save_fpath = os.path.join(model_path, "anova", "univar_fea_selector.pkl")
    #     with open(save_fpath, "wb") as f:
    #         pickle.dump(univar_fea_selector, f)

    # # Cross validation
    # if CROSS_VALIDATION:
    #     cross_validation(train_X, train_y, model, params=determined_params, cv=10)

    # # Train the final model and predict with test set
    # clf = train_model(model, train_X, train_y, test_X=None, test_y=None, params=determined_params, grid_search=False, param_grid=param_grid, scoring=None)
    # save_fpath = os.path.join(model_save_path, model_name + "_Normal.pkl")
    # with open(save_fpath, "wb") as f:
    #     pickle.dump(clf, f)
    # test_y_pred = clf.predict(test_X)
    # print("=== {} trained. Attributes of RF:".format(model))
    # # pprint({attr: clf.__getattribute__(attr) for attr in dir(clf)}) # print the attributes of the trained classifier

    # # Evaluate
    # acc_score = accuracy_score(test_y, test_y_pred)
    # confusion_mat = confusion_matrix(test_y_pred, test_y, confusion_mat_only=True)
    # dr = detection_rate(test_y_pred, test_y)
    # far = false_alarm_rate(test_y_pred, test_y)
    # feature_weights = list(zip(train_X.columns, clf.feature_importances_))
    # feature_weights.sort(key=lambda x: x[1], reverse=True)
    # print("=== Results of {}:".format(model))
    # print("Final acc:", acc_score)
    # # print("Final score:", score)
    # print("Final detection rate:", dr)
    # print("Final false alarm rate:", far)
    # print("Final confusion_matrix:", "\n", confusion_mat)
    # # pprint(feature_weights)
    
    # # 对Normal集Kmeans聚类，Normal集要经过ANOVA特征选择
    # if KMEANS:
    #     print("=== Training Kmeans")
    #     kmeans = KMeans(n_clusters=kmeans_cent_nums["Normal"])
    #     normal_train_X = normal_train_df.drop("attack_type", axis=1).values
    #     if ANOVA:
    #         normal_train_X = univar_fea_selector.transform(normal_train_X)
    #     kmeans.fit(normal_train_X)
    #     centroids = kmeans.cluster_centers_
    #     # print(" -- Calculating silh_coef")
    #     # slih_coef = silhouette_coef(normal_train_X, kmeans.labels_)
    #     # print("Kmeans silh_coef", slih_coef)
    #     save_fpath = os.path.join(model_path, "kmeans", "Normal_kmeans_centroids.pkl")
    #     with open(save_fpath, "wb") as f:
    #         pickle.dump(centroids, f)


    ###############################################
    # 对不同攻击类型构建二分类训练集，训练二分类模型 #
    ###############################################
    # labels = ['1_DOS', '2_Probe', '3_R2L', '4_U2R']
    # labels = ['3_R2L', '4_U2R']
    labels = ['2_Probe']
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
        attack_test_fname = label + "_test.csv"
        attack_test_fpath = os.path.join(prep_path, attack_test_fname)
        attack_test_df = pd.read_csv(attack_test_fpath)
        attack_type = label[2:]

        if DOWNSAMPLE:
            normal_train_fname = "0_Normal_downsampled_by_" + label + ".csv"
            normal_train_fpath = os.path.join(filepaths.TRAIN_DOWNSAMPLE_FOLDER_PATH, normal_train_fname)
            try:
                normal_train_df = pd.read_csv(normal_train_fpath)
            except FileNotFoundError:
                normal_train_fpath = os.path.join(filepaths.PREPROCESS_PATH, '0_Normal.csv')
                normal_train_df = pd.read_csv(normal_train_fpath)
        
        print("=== {}: normal samples: {}, attack samples: {}".format(attack_type, len(normal_train_df), len(attack_train_df)))

        train_df = pd.concat([normal_train_df, attack_train_df], ignore_index=True)
        test_df = pd.concat([normal_test_df, attack_test_df], ignore_index=True)
        train_df = shuffle(train_df)
        train_df = train_df.reset_index(drop=True)
        train_X, train_y = data_label_split(train_df)
        test_X, test_y = data_label_split(test_df)

        # Univariate feature selection
        if ANOVA:
            train_X, test_X = univar_fea_selector.transform(train_X), univar_fea_selector.transform(test_X)

        # Cross validation
        if CROSS_VALIDATION:
            cross_validation(train_X, train_y, model, params=determined_params, cv=10)

        # Train the final model and predict with test set
        clf = train_model(model, train_X, train_y, test_X=None, test_y=None, params=determined_params, grid_search=False, param_grid=param_grid, scoring=None)
        save_fpath = os.path.join(model_save_path, model_name + "_" + attack_type + ".pkl")
        with open(save_fpath, "wb") as f:
            pickle.dump(clf, f)
        test_y_pred = clf.predict(test_X)
        print("=== {} trained. Attributes of RF:".format(model))
        # pprint({attr: clf.__getattribute__(attr) for attr in dir(clf)}) # print the attributes of the trained classifier

        # Evaluate
        acc_score = accuracy_score(test_y, test_y_pred)
        confusion_mat = confusion_matrix(test_y_pred, test_y, confusion_mat_only=True)
        dr = detection_rate(test_y_pred, test_y)
        far = false_alarm_rate(test_y_pred, test_y)
        # feature_weights = list(zip(train_X.columns, clf.feature_importances_))
        # feature_weights.sort(key=lambda x: x[1], reverse=True)
        print("=== Results of {}:".format(model))
        print("Final acc:", acc_score)
        # print("Final score:", score)
        print("Final detection rate:", dr)
        print("Final false alarm rate:", far)
        print("Final confusion_matrix:", "\n", confusion_mat)
        # pprint(feature_weights)

        # Kmeans聚类
        if KMEANS:
            print("=== Training Kmeans")
            kmeans = KMeans(n_clusters=kmeans_cent_nums[attack_type])
            attack_train_X = attack_train_df.drop("attack_type", axis=1).values
            if ANOVA:
                attack_train_X = univar_fea_selector.transform(attack_train_X)
            kmeans.fit(attack_train_X)
            centroids = kmeans.cluster_centers_
            # print(" -- Calculating silh_coef")
            # slih_coef = silhouette_coef(attack_train_X, kmeans.labels_)
            # print("Kmeans silh_coef", slih_coef)
            save_fpath = os.path.join(model_path, "kmeans", attack_type + "_kmeans_centroids.pkl")
            with open(save_fpath, "wb") as f:
                pickle.dump(centroids, f)