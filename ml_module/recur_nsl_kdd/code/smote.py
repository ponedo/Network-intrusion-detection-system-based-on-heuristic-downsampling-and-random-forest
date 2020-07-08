import utils.filepaths as filepaths
import os, random, getopt, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def eu_dist(x, y):
    return ((y - x) ** 2).sum() ** 0.5


def eu_dist_batch(x, y):
    return ((y - x) ** 2).sum(axis=1) ** 0.5


def get_key_minority_set(maj_df, min_df, M):
    maj_arr = maj_df.drop('attack_type', axis=1).values
    min_arr = min_df.drop('attack_type', axis=1).values
    min_sample_n = min_arr.shape[0]
    maj_sample_n = maj_arr.shape[0]

    # 计算min_arr内部各个点的距离
    print("------ 计算min_arr内部各个点的距离")
    min2min_dist_mat = np.zeros((min_sample_n, min_sample_n))
    for i, min_point in enumerate(min_arr):
        min2min_dist_mat[i, :] = eu_dist_batch(min_point, min_arr)

    # 计算min_arr各个点到maj_arr中各个点的距离
    print("------ 计算min_arr各个点到maj_arr中各个点的距离")
    min2maj_dist_mat = np.zeros((min_sample_n, maj_sample_n))
    for i, min_point in enumerate(min_arr):
        min2maj_dist_mat[i, :] = eu_dist_batch(min_point, maj_arr)
    
    key_set_index = []
    # 对每个min_point，检查M个最近邻的标签情况
    print("------ 对每个min_point，检查M个最近邻的标签情况")
    for i in range(min_sample_n):
        dist_arr_min = [(d, 1) for d in min2min_dist_mat[i, :]]
        dist_arr_maj = [(d, 0) for d in min2maj_dist_mat[i, :]]
        dist_arr = dist_arr_min + dist_arr_maj
        dist_arr.sort(key=lambda x: x[0])
        m = 0
        for _, label in dist_arr[1 : M + 1]:
            if label == 0:
                m += 1
        if m >= M / 2 and m < M:
            key_set_index.append(i)
    
    print('------', len(key_set_index), min_sample_n)
    return min_df.loc[key_set_index, :], min2min_dist_mat[key_set_index, :][:, key_set_index]


def smote(min_df, K, S, dist_mat=None, msmote=False):
    '''
        对小数量的样本集进行过采样
        返回过采样后的新数据集
    '''
    def interpolate(center, knn, hyperball=False):
        '''
            利用k个最近邻样本点生成新样本
            先用线性试一下
            param.center: 生成新样本的基准点
            param.knn: 按距离升序排列的基准点的k个近邻点
        '''
        new_samples = []
        if hyperball:
            # 找到最远近邻点，以其为超球体边界
            remote = knn[-1, :]
            r = eu_dist(remote, center)
            dim = remote.shape[0]
            for _ in range(S):
                # 在超球体边界上随机找一点
                sigmas = []
                norm_factor = 0
                for _ in range(dim):
                    sigma = random.random()
                    sigmas.append(sigma)
                    norm_factor += sigma ** 2
                norm_factor = norm_factor ** 0.5
                sigmas = [sigma / norm_factor for sigma in sigmas]
                border_point = np.array([x + sigma * r for x, sigma in zip(center, sigmas)])

                # 插值
                p = random.random()
                new_sample = (1 - p) * center + p * (border_point)
                new_samples.append(new_sample)

        else:
            # 线性插值
            rand_samples = []
            for _ in range(S):
                rand_samples.append(random.choice(list(knn)))
            for p_rand in rand_samples:
                p = random.random()
                new_sample = (1 - p) * center + p * p_rand
                new_samples.append(new_sample)
        return np.array(new_samples)

    min_arr = min_df.drop('attack_type', axis=1).values
    sample_n, dim = min_arr.shape[0], min_arr.shape[1]

    if dist_mat is None:
        # 计算min_arr内部各个点的距离
        dist_mat = np.zeros((sample_n, sample_n))
        for i, min_point in enumerate(min_arr):
            dist_mat[i, :] = eu_dist_batch(min_point, min_arr)
    
    new_samples = np.zeros((sample_n * (S+1), dim))
    new_samples[sample_n*S:, :] = min_arr
    # 对每个样本点，找到K个最近邻knn，利用knn插值生成新样本
    print("------ 对每个样本点，找到K个最近邻knn，利用knn插值生成新样本")
    for i, min_point in enumerate(min_arr):
        # 找到knn
        dist_arr = [(d, j) for j, d in enumerate(dist_mat[i, :])]
        dist_arr.sort(key=lambda x: x[0])
        knn_index = [j for _, j in dist_arr[1:K+1]]
        knn = min_arr[knn_index, :]
        new_samples_i = interpolate(min_point, knn, hyperball=msmote)
        new_samples[i*S:(i+1)*S, :] = new_samples_i
    
    # 将new_samples数组转化为DataFrame
    new_samples_df = pd.DataFrame(new_samples)
    label = min_df['attack_type'].values[0]
    new_samples_df['attack_type'] = label
    new_samples_df.columns = min_df.columns
    return new_samples_df


if __name__ == "__main__":

    opts, args = getopt.getopt(sys.argv[1:], "pa:", ["pca"])
    opts = dict(opts)
    
    normal_df = pd.read_csv(os.path.join(filepaths.PREPROCESS_PATH, '0_Normal.csv'))
    # labels = ['4_U2R']
    labels = ['3_R2L', '4_U2R']
    # labels = ['1_DOS', '2_Probe', '3_R2L', '4_U2R']

    print("========= Oversampling =========")
    for label in labels:
        print("--------- {} ---------".format(label))
        fpath = os.path.join(filepaths.PREPROCESS_PATH, '{}.csv'.format(label))
        sub_df = pd.read_csv(fpath)
        M = 100 if label == '3_R2L' else 5 # 搜索关键点时考虑的周围群体范围
        print("--- Generating key set...")
        key_set_df, dist_mat = get_key_minority_set(normal_df, sub_df, M)
        K = 20 if label == '3_R2L' else 5 # K近邻范围
        S = 10 if label == '3_R2L' else 430 # 扩大倍数
        #K > S
        print("--- smote...")
        BORDERLINE = True
        if BORDERLINE:
            oversampled_df = smote(key_set_df, K, S, dist_mat, msmote=True)
        else:
            oversampled_df = smote(sub_df, K, S, dist_mat=None, msmote=True)
        print("Orig samples: {}, then samples: {}".format(len(sub_df), len(oversampled_df)))
        fpath_new = os.path.join(filepaths.TRAIN_SMOTE_FOLDER_PATH, '{}_oversampled.csv'.format(label))
        oversampled_df.to_csv(fpath_new, index=False)


    if "--pca" in opts or "-p" in opts:
        print("========= PCA PLOTTING =========")

        train_path = filepaths.TRAIN_PATH
        img_path = filepaths.IMG_PATH
        train_df = pd.read_csv(train_path)
        fig, axes = plt.subplots(6, 3, figsize=(15, 24))

        # 读原正负样本数据
        X = train_df.drop('attack_type', axis=1).values
        c = 1
        y = [c for _ in range(len(X))]

        # 用原正负样本数据训练pca
        pca = PCA(n_components=2)
        X_new = pca.fit_transform(X)
        X_all = X_new.copy()
        y_all = y

        # 绘制过采样数据
        smote_data_dir = filepaths.TRAIN_SMOTE_FOLDER_PATH
        for fname in os.listdir(smote_data_dir):
            fpath = os.path.join(smote_data_dir, fname)
            oversampled_df = pd.read_csv(fpath)
            X = oversampled_df.drop('attack_type', axis=1).values
            X_new = pca.transform(X)

            label_num = int(fname[0])
            plt.subplot(6, 3, label_num * 3 + 1)
            plt.xlim(-2.5, 3)
            plt.ylim(-2, 3.5)
            plt.title(fname)
            plt.scatter(X_new[:, 0], X_new[:, 1], linewidths=0.01, marker='o')

        # 绘制原数据
        labels = ['0_Normal', '1_DOS', '2_Probe', '3_R2L', '4_U2R']
        prep_path = filepaths.PREPROCESS_PATH
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
        plt.savefig(os.path.join(img_path, "smote.png"))
