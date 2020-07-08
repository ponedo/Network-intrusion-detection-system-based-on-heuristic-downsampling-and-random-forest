import os

TRAIN_TXT_PATH = os.path.join("../data/raw_data", "KDDTrain+.txt")
TEST_TXT_PATH = os.path.join("../data/raw_data", "KDDTest+.txt")
PREPROCESS_PATH = os.path.join("../data", "preprocessed_data")
TRAIN_PATH = os.path.join("../data/preprocessed_data", "train.csv")
TEST_PATH = os.path.join("../data/preprocessed_data", "test.csv")
TRAIN_BINARY_PATH = os.path.join("../data/preprocessed_data", "train_binary.csv")
TEST_BINARY_PATH = os.path.join("../data/preprocessed_data", "test_binary.csv")
TRAIN_LMDRT_BINARY_PATH = os.path.join("../data/lmdrt/preprocessed_data", "train_lmdrt_binary.csv")
TEST_LMDRT_BINARY_PATH = os.path.join("../data/lmdrt/preprocessed_data", "test_lmdrt_binary.csv")

TRAIN_SMOTE_FOLDER_PATH = os.path.join("../data/preprocessed_data", "smote")
TRAIN_DOWNSAMPLE_FOLDER_PATH = os.path.join("../data/preprocessed_data", "downsample")
IMG_PATH = os.path.join("../", "img")
FRAME_MODEL_PATH = os.path.join("../", "models")
