import utils.filepaths as filepaths
import pandas as pd
import os, sys, getopt, time
from pprint import pprint
from nn import DNNModel
from sklearn.decomposition import KernelPCA
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

from utils.train_utils import *
from utils.metric import *


SMOTE = False
OVERSAMPLE = False and SMOTE
DOWNSAMPLE = False
BINARY = False
ANOVA, ANOVA_PERCENTILE = False, 70
KPCA, KPCA_D = False, 30
GRID_SEARCH = False
RFE, RFE_GRID_SEARCH = False, False
CROSS_VALIDATION = False


def score_func(estimator, X, true_y):
    pred_y = estimator.predict(X)
    return confusion_matrix(pred_y, true_y, score_only=True)


def train_and_test_model(train_df, test_df, TIME=1):

    test_X, test_y = data_label_split(test_df)
    test_y_pred = np.array([])
    # test_y_proba_pred = np.reshape(np.array([]), (-1, 5))
    test_y = np.array(TIME * test_y.tolist())
    train_time, test_time = [], []

    for _ in range(TIME):

        # shuffle
        train_df = shuffle(train_df)
        train_df = train_df.reset_index(drop=True)

        # Split X and y
        train_X, train_y = data_label_split(train_df)

        # Univariate feature selection
        if ANOVA:
            train_X, test_X, univar_fea_selector = univar_feat_select(train_X, test_X, train_y, method=f_classif, percentile=ANOVA_PERCENTILE)

        # KPCA transform
        if KPCA:
            kpca = KernelPCA(kernel='rbf', n_components=KPCA_D)
            train_X = kpca.fit_transform(train_X)
            test_X = kpca.transform(test_X)

        # Recursive Feature Elimilation
        if RFE:
            train_X = rfe(model, train_X, train_y, test_X, test_y, 15, determined_params, RFE_GRID_SEARCH, param_grid, score_func)

        # Cross validation
        if CROSS_VALIDATION:
            cross_validation(train_X, train_y, model, params=determined_params, cv=10)

        # Train the final model and predict with test set
        t1 = time.time()
        clf = train_model(model, train_X, train_y, test_X, test_y, determined_params, GRID_SEARCH, param_grid, score_func)
        t2 = time.time()
        test_y_pred = np.append(test_y_pred, clf.predict(test_X))
        t3 = time.time()
        # test_y_proba_pred = np.append(test_y_proba_pred, clf.predict_proba(test_X), axis=0)
        train_time.append(t2 - t1)
        test_time.append(t3 - t2)
        # print("=== {} trained. Attributes of RF:".format(model))
        # pprint({attr: clf.__getattribute__(attr) for attr in dir(clf)}) # print the attributes of the trained classifier

    # Calculate metrics
    print("=== Train time: {}, test time: {}".format(sum(train_time) / len(train_time), sum(test_time) / len(test_time)))
    acc_score = accuracy_score(test_y, test_y_pred)
    confusion_mat= confusion_matrix(test_y_pred, test_y, confusion_mat_only=True)
    bin_metric = binary_metric(test_y_pred, test_y)
    # pos_proba = [1 - p[0] for p in test_y_proba_pred]
    # auc_value = roc_auc(pos_proba, test_y)
    # dr = detection_rate(test_y_pred, test_y)
    # far = false_alarm_rate(test_y_pred, test_y)
    # feature_weights = list(zip(train_X.columns, clf.feature_importances_))
    # feature_weights.sort(key=lambda x: x[1], reverse=True)
    print("=== Results of {}:".format(model))
    print("Final acc:", acc_score)
    print("Final binary metric:", bin_metric)
    # print("Final AUC:", auc_value)
    # print("Final score:", score)
    # print("Final detection rate:", dr)
    # print("Final false alarm rate:", far)
    print("Final confusion_matrix:", "\n", confusion_mat)
    # pprint(feature_weights)


if __name__ == "__main__":
    # Parse cmd args
    """
    --model -m [svm|rf|gdbt]
    --smote -s
    --downsample -d
    --binary -b
    --anova -a
    --kpca -k
    --grid-search -g
    --cv -v
    --rfe -r
    """
    opts, args = getopt.getopt(
        sys.argv[1:], "m:sdb:akgvr", 
        ["model=", "smote", "downsample", "binary=", "anova", "kpca", "grid-search", "cv", "rfe"])
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
    BINARY = BINARY or "--binary" in opts or "-b" in opts
    if BINARY:
        test_label = opts["--binary"] or opts["-b"]
    print("BINARY:", BINARY)
    ANOVA = ANOVA or "--anova" in opts or "-a" in opts
    print("ANOVA:", ANOVA)
    KPCA = KPCA or "--kpca" in opts or "-k" in opts
    print("KPCA:", KPCA)

    # Read input model
    if model_name == "svm":
        model = SVC
        determined_params = {
            'kernel': 'rbf',
            'gamma': 'auto'
        }
        param_grid = {
            'kernel': ['rbf'], 
            'C': [0.01, 0.05, 0.1, 0.5, 1, 5], # 松弛变量
            # 'gamma': [0.01]
        }
    elif model_name == "rf":
        model = RandomForestClassifier
        determined_params = {
            'n_estimators': 50, 
            'n_jobs': 2, 
            'criterion': 'entropy', 
            # 'min_impurity_split': 0, 
        }
        param_grid = {
            'criterion': ['gini', 'entropy'], 
            'n_estimators': range(10, 100, 10), 
            'min_samples_split': range(2, 10), 
            'min_samples_leaf': range(1, 5), 
            # 'max_depth': [None], 
            # 'max_leaf_nodes': [None]
        }
    elif model_name == "gbdt":
        model = GradientBoostingClassifier
        determined_params = {
            'n_estimators': 100, 
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


    # Read dataset
    if SMOTE and DOWNSAMPLE:
        if BINARY:
            fnames = ["0_Normal_downsampled", test_label + '_oversampled']
        else:
            fnames = ['0_Normal_downsampled', '1_DOS_downsampled', '2_Probe_downsampled', '3_R2L', '4_U2R', '3_R2L_oversampled', '4_U2R_oversampled']
            # fnames = ['0_Normal_downsampled', '1_DOS', '2_Probe', '3_R2L', '4_U2R', '3_R2L_oversampled', '4_U2R_oversampled']
    elif SMOTE and not DOWNSAMPLE:
        if BINARY:
            fnames = ['0_Normal', test_label + '_oversampled']
        else:
            fnames = ['0_Normal', '1_DOS', '2_Probe', '3_R2L', '4_U2R', '3_R2L_oversampled', '4_U2R_oversampled']
    elif not SMOTE and DOWNSAMPLE:
        if BINARY:
            fnames = ['0_Normal_downsampled', test_label]
        else:
            fnames = ['0_Normal_downsampled', '1_DOS_downsampled', '2_Probe_downsampled', '3_R2L', '4_U2R']
            # fnames = ['0_Normal_downsampled', '2_Probe_downsampled', '1_DOS', '3_R2L', '4_U2R']
            # fnames = ['0_Normal_downsampled', '1_DOS_downsampled', '2_Probe', '3_R2L', '4_U2R']
            # fnames = ['0_Normal_downsampled', '1_DOS', '2_Probe', '3_R2L', '4_U2R'] # 提升bin acc dr f1, 降低bin prec far
    else:
        if BINARY:
            fnames = ['0_Normal', test_label]
        else:
            fnames = ['train']

    train_dfs = []
    for fname in fnames:
        print(fname)
        if "oversampled" in fname:
            fpath = os.path.join(filepaths.TRAIN_SMOTE_FOLDER_PATH, '{}.csv'.format(fname))
        elif "downsampled" in fname:
            fpath = os.path.join(filepaths.TRAIN_DOWNSAMPLE_FOLDER_PATH, '{}.csv'.format(fname))
        else:
            fpath = os.path.join(filepaths.PREPROCESS_PATH, '{}.csv'.format(fname))
        train_dfs.append(pd.read_csv(fpath))
    train_df = pd.concat(train_dfs, ignore_index=True)

    
    if BINARY:
        attack_fpath = os.path.join(filepaths.PREPROCESS_PATH, test_label + "_test.csv")
        test_attack_df = pd.read_csv(attack_fpath)
        normal_fpath = os.path.join(filepaths.PREPROCESS_PATH, "0_Normal_test.csv")
        test_normal_df = pd.read_csv(normal_fpath)
        test_df = pd.concat([test_attack_df, test_normal_df], ignore_index=True)
    else:
        test_df = pd.read_csv(filepaths.TEST_PATH)

    train_and_test_model(train_df, test_df, TIME=10)