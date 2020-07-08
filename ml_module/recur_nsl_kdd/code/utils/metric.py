"""
    使用特定的cost matrix来计算分类器的分数
    具体的算法见nsl report 2016
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from .filepaths import IMG_PATH
from sklearn.metrics import roc_curve, auc

cost_matrix = np.array([
    [0, 1, 2, 2, 2], 
    [1, 0, 2, 2, 2], 
    [2, 1, 0, 2, 2], 
    [3, 2, 2, 0, 2], 
    [4, 2, 2, 2, 0]
])


def confusion_matrix(pred_y, true_y, score_only=False, confusion_mat_only=False):
    """
    竖直维为真实，横向维为预测
    """
    if not isinstance(pred_y, list) or not isinstance(true_y, list):
        pred_y, true_y = pred_y.tolist(), true_y.tolist()
    rows, cols = set(true_y), set(pred_y)
    labels = list(rows | cols)
    labels.sort()
    label2code, code = {}, 0
    for label in labels:
        label2code[label] = code
        code += 1
    confusion_matrix = np.zeros((len(labels), len(labels)))
    for pred, true in zip(pred_y, true_y):
        confusion_matrix[label2code[true], label2code[pred]] += 1

    # calculate precision and recall
    precision, recall = {}, {}
    for i, label in enumerate(labels):
        prec = confusion_matrix[i, i] / np.sum(confusion_matrix[:, i])
        rec = confusion_matrix[i, i] / np.sum(confusion_matrix[i, :])
        precision[label] = prec
        recall[label] = rec

    confusion_matrix_df = pd.DataFrame(confusion_matrix, index=labels, columns=labels)
    precision = pd.Series(precision, name='precision')
    recall['precision'] = None
    recall = pd.Series(recall, name='recall')
    confusion_matrix_df = confusion_matrix_df.append(precision)
    confusion_matrix_df = pd.concat([confusion_matrix_df, recall], axis=1)

    # calcucate score based on cost matrix
    if confusion_mat_only:
        return confusion_matrix_df
    
    cost_mat = cost_matrix * confusion_matrix
    cost = sum(sum(cost_mat))
    correct = 0
    assert confusion_matrix.shape[0] == confusion_matrix.shape[1]
    for i in range(confusion_matrix.shape[0]):
        correct += confusion_matrix[i, i]
    score = correct / (correct + cost)

    if score_only:
        # print(confusion_matrix_df)
        return score
    return confusion_matrix_df, score


def detection_rate(pred_y, true_y):
    assert len(pred_y) == len(true_y)
    tp, p = 0, 0
    for pred, label in zip(pred_y, true_y):
        if not label == "0_Normal": # normal label should be Normal_0
            p += 1
            if pred == label:
                tp += 1
    dr = tp / p
    return dr


def false_alarm_rate(pred_y, true_y):
    assert len(pred_y) == len(true_y)
    fp, n = 0, 0
    for pred, label in zip(pred_y, true_y):
        if label == "0_Normal": # normal label should be Normal_0
            n += 1
            if not pred == label:
                fp += 1
    far = fp / n
    return far

    
def eu_dist(x, y):
    return ((y - x) ** 2).sum() ** 0.5


def eu_dist_batch(x, y):
    return ((y - x) ** 2).sum(axis=1) ** 0.5


def silhouette_coef(X, Y):
    """
    计算轮廓系数
    param X: 样本点
    param y: 聚类标签
    """
    def a(x, X_same):
        """
        计算样本点x的簇内不相似度
        param x: 样本点
        param X: 与x同簇的簇
        """
        s = 0
        for x_i in X_same:
            d = eu_dist(x, x_i)
            s += d
        return s / (len(X_same) - 1)

    def b(x, X_other):
        """
        计算样本点x的簇内不相似度
        param x: 样本点
        param X: 与x同簇的簇
        """
        s = 0
        for x_i in X_other:
            d = eu_dist(x, x_i)
            s += d
        return s / len(X_other)

    label_set = set(Y)
    clusters = {
        label: X[Y==label, :] for label in label_set
    }

    s = 0
    i = 0
    for x, y in zip(X, Y):
        i += 1
        if i % 5 == 0:
            print(i)
        X_same = X[Y == y, :]
        a_x = a(x, X_same)
        b_x = float("inf")
        for _, X_other in filter(lambda x: (not x[0] == y), clusters.items()):            
            b_x = min(b_x, b(x, X_other))
        if a_x < b_x:
            s_x = 1 - a_x / b_x
        elif a_x == b_x:
            s_x = 0
        elif a_x > b_x:
            s_x = b_x / a_x - 1
        s += s_x
    return s / len(Y)


def binary_metric(pred_y, true_y):
    """
    仅将标签视为0_Normal和其他异常类
    返回一个字典，包含acc, f1
    positive: 异常
    negative: 正常
    """
    assert len(pred_y) == len(true_y)

    tp, tn, fp, fn = 0, 0, 0, 0
    for pred, true in zip(pred_y, true_y):
        if pred == "0_Normal" and true == "0_Normal":
            tn += 1
        elif pred == "0_Normal" and not true == "0_Normal":
            fn += 1
        elif not pred == "0_Normal" and true == "0_Normal":
            fp += 1
        elif not pred == "0_Normal" and not true == "0_Normal":
            tp += 1
    acc = (tp + tn) / (tp + tn + fp + fn)
    dr = tp / (tp + fn)
    prec = tp / (tp + fp)
    far = fp / (fp + tn)
    f1 = 2 * prec * dr / (prec + dr)
    return {
        "binary accuracy": acc, 
        "binary detection rate": dr, 
        "binary precision": prec, 
        "binary false alarm rate": far, 
        "binary f1 score": f1
    }


def roc_auc(pred_y_score, true_y):
    def merge_labels(y):
        return np.array([0 if p =="0_Normal" else 1 for p in y])
    true_y = merge_labels(true_y)
    fpr, tpr, thresholds = roc_curve(true_y, pred_y_score)
    auc_value = auc(fpr, tpr)
    plt.plot(fpr, tpr, marker='o')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(os.path.join(IMG_PATH, "roc.png"))
    return auc_value