from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
import os
import utils.filepaths as filepaths
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

n_components = 2
KPCA = False


if __name__ == "__main__":

    train_path = filepaths.TRAIN_PATH
    img_path = filepaths.IMG_PATH
    train_df = pd.read_csv(train_path)
    fig, axes = plt.subplots(8, 2, figsize=(15, 24))

    # 读原正负样本数据
    X = train_df.drop('attack_type', axis=1).values
    c = 1
    y = [c for _ in range(len(X))]

    # 用原正负样本数据训练pca
    if KPCA:
        pca = KernelPCA(n_components=n_components, kernel="rbf")
        X_new = pca.fit_transform(X)
    else:
        pca = PCA(n_components=n_components)
        X_new = pca.fit_transform(X)
    X_all = X_new.copy()
    y_all = y

    # 设置坐标轴范围
    plt.subplot(6, 3, 1)
    plt.xlim(-2.5, 3)
    plt.ylim(-2, 3.5)

    # 绘制原正负样本
    plt.scatter(X_new[:, 0], X_new[:, 1], c=y, linewidths=0.01, marker='o')
    plt.title(train_path)
    # plt.savefig(os.path.join(img_path, "Raw_samples.png"))

    smote_data_dir = filepaths.TRAIN_SMOTE_FOLDER_PATH
    for i, fname in enumerate(os.listdir(smote_data_dir)):
        # 读插值数据
        fpath = os.path.join(smote_data_dir, fname)
        df = pd.read_csv(fpath)

        # 读原正负样本数据
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
        plt.subplot(6, 3, i+2)
        plt.scatter(X_new[:, 0], X_new[:, 1], c=y, linewidths=0.01, marker='o')
        plt.xlim(-2.5, 3)
        plt.ylim(-2, 3.5)
        plt.title(fname)
        # plt.savefig(os.path.join(img_path, fname[:-4] + ".png"))

    plt.subplot(6, 3, 17)
    plt.xlim(-2.5, 3)
    plt.ylim(-2, 3.5)
    plt.scatter(X_all[:, 0], X_all[:, 1], c=y_all, linewidths=0.01, marker='o')
    plt.savefig(os.path.join(img_path, "all.png"))
