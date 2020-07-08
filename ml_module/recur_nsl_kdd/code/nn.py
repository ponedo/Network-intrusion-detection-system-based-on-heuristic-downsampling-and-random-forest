import os, pickle
import numpy as np
import pandas as pd
import utils.filepaths as filepaths
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.core import Dropout
from keras import optimizers
from keras.utils import to_categorical
from sklearn.utils import shuffle

from utils.train_utils import *
from utils.metric import *


SMOTE = False
ANOVA, ANOVA_PERCENTILE = False, 15
CROSS_VALIDATION = False

INPUT_DIM = None
CLASS_NUM = 5
DNN_NEURAL_NUM = [128, 128, 48, 16]
DNN_ACTIVATE = ["relu", "tanh", "tanh", "relu"]
DROPOUT_P = 0.15
EPOCH = 4
BATCH_SIZE = 32



# def score_func(estimator, X, true_y):
#     pred_y = estimator.predict(X)
#     return confusion_matrix(pred_y, true_y, confusion_mat_only=True)


class NNModel():
    def __init__(self):
        self.model = self._buildModel()

    def fit(self, X, y):
        y = self._one_hot_encode(y)
        self.model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCH)

    def predict(self, X):
        y = self.model.predict(X)
        y = self._one_hot_decode(y)
        return y
    
    def predict_proba(self, X):
        y = self.model.predict(X)
        return y

    def _buildModel(self):
        return None

    def _one_hot_encode(self, Y):
        def _one_hot(i, n):
            arr = np.zeros((n, ))
            arr[i] = 1.
            return arr
        labels = np.unique(Y)
        n = len(labels)
        self._one_hot_encoder = {label: i for i, label in enumerate(labels)}
        self._one_hot_decoder = {i: label for label, i in self._one_hot_encoder.items()}
        new_Y = [_one_hot(self._one_hot_encoder[y], n) for y in Y]
        return np.array(new_Y)

    def _one_hot_decode(self, Y):
        Y = np.argmax(Y, axis=1)
        return np.array([self._one_hot_decoder[y] for y in Y])
        


class DNNModel(NNModel):
    def __init__(self):
        self.model =  self._buildModel()

    def _buildModel(self):
        model = Sequential()
        for neural_num, activation in zip(DNN_NEURAL_NUM, DNN_ACTIVATE):
            model.add(Dense(neural_num, activation=activation))
            model.add(Dropout(DROPOUT_P))
        model.add(Dense(CLASS_NUM, activation="softmax"))

        rms = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0) 
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        ada = optimizers.Adagrad(lr=0.01, epsilon=1e-06)

        model.compile(
            # optimizer='rmsprop',
            optimizer=ada, 
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model


if __name__ == "__main__":
    # 读训练数据集
    model = DNNModel

    # Read dataset
    if SMOTE:
        fnames = ['0_Normal', '1_DOS', '2_Probe', '3_R2L', '4_U2R', '3_R2L_oversampled', '4_U2R_oversampled']
        train_dfs = []
        for fname in fnames:
            fpath = os.path.join(filepaths.TRAIN_SMOTE_FOLDER_PATH, '{}.csv'.format(fname))
            train_dfs.append(pd.read_csv(fpath))
        train_df = pd.concat(train_dfs, ignore_index=True)
    else:
        train_df = pd.read_csv(filepaths.TRAIN_PATH)
    train_df = shuffle(train_df)
    train_df = train_df.reset_index(drop=True)
    test_df = pd.read_csv(filepaths.TEST_PATH)

    # Split X and y
    train_X, train_y = data_label_split(train_df)
    test_X, test_y = data_label_split(test_df)

    # Univariate feature selection
    model_path = filepaths.FRAME_MODEL_PATH
    if ANOVA:
        univar_fea_select_fpath = os.path.join(model_path, "anova", "univar_fea_selector.pkl")
        with open(univar_fea_select_fpath, "rb") as f:
            univar_fea_selector = pickle.load(f)
        train_X = univar_fea_selector.transform(train_X)
        test_X = univar_fea_selector.transform(test_X)

    # Cross validation
    if CROSS_VALIDATION:
        cross_validation(train_X, train_y, model, params={}, cv=10)

    # Train the final model and predict with test set
    clf = train_model(model, train_X, train_y, test_X=None, test_y=None, params={}, grid_search=False, param_grid={}, scoring=None)
    test_y_pred = clf.predict(test_X)
    print("=== {} trained. Attributes of RF:".format(model))
    # pprint({attr: clf.__getattribute__(attr) for attr in dir(clf)}) # print the attributes of the trained classifier

    # Evaluate
    acc_score = accuracy_score(test_y, test_y_pred)
    confusion_mat, score = confusion_matrix(test_y_pred, test_y)
    bin_metric = binary_metric(test_y_pred, test_y)
    # dr = detection_rate(test_y_pred, test_y)
    # far = false_alarm_rate(test_y_pred, test_y)
    print("=== Results of {}:".format(model))
    print("Final acc:", acc_score)
    print("Final binary metric:", bin_metric)
    print("Final score:", score)
    # print("Final detection rate:", dr)
    # print("Final false alarm rate:", far)
    print("Final confusion_matrix:", "\n", confusion_mat)
    # pprint(feature_weights)

