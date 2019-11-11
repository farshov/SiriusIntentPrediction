import pandas as pd
from sklearn.model_selection import train_test_split
from data_load.loading import load_from_json, load_from_csv
from data_load.loading import preprocess_labels
from feature_extraction.content_feature_extractor import Content
from feature_extraction.structural_feature_extractor import Structural
from models.ml_models import MLModels
from feature_extraction.sentiment_feature_extractor import Sentiment
from sklearn.model_selection import KFold
import numpy as np
import json
from quality_metrics.stat_tests import perform_stat_tests
from feature_extraction.build_dataframe import build_and_write_dataset
import warnings
warnings.filterwarnings("ignore")

labels = ['OQ', 'RQ', 'CQ', 'FD', 'FQ', 'IR', 'PA', 'PF', 'NF', 'GG', 'JK', 'O']
label_dict = dict(list(zip(labels, range(0, len(labels)))))

ml = MLModels()
models = ["KNN", "NaiveBayes", "SVM", "RandomForest", "AdaBoost"]

df = pd.read_csv("data/msdialogue/train_features.tsv", header=None, delimiter="\t")
arr = df.iloc[:, 1]
X_train_theirs = pd.DataFrame(np.array(list(map(lambda x: x.split(), arr))).astype("float32"))
X_train_theirs = X_train_theirs.drop([0,1], axis=1)
y_train = preprocess_labels(list(df.iloc[:, 0]), label_dict)

df = pd.read_csv("data/msdialogue/test_features.tsv", header=None, delimiter="\t")
arr = df.iloc[:, 1]
X_test_theirs = pd.DataFrame(np.array(list(map(lambda x: x.split(), arr))).astype("float32"))
X_test_theirs = X_test_theirs.drop([0,1], axis=1)
y_test = preprocess_labels(list(df.iloc[:, 0]), label_dict)

df = pd.read_csv("data/msdialogue/token_train.csv")
X_train_our = df.drop(["initial_utterance_similarity","dialog_similarity"], axis=1)
df = pd.read_csv("data/msdialogue/token_test.csv")
X_test_our = df.drop(["initial_utterance_similarity","dialog_similarity"], axis=1)

predictions_theirs = ml.get_basic_model_predictions(X_train_theirs, y_train, X_test_theirs)
predictions_our = ml.get_basic_model_predictions(X_train_our, y_train, X_test_our)

for i, model in enumerate(models):
    print("MODEL: {}".format(model))
    perform_stat_tests(predictions_our[i], predictions_theirs[i], y_test)
    print("\n")

# build_and_write_dataset("data/msdialogue/test_data.json", "data/msdialogue/test_data.csv")
# with open("data/msdialogue/train_data.json") as f:
#     X = json.load(f)
# content_features = Content()
# structural_features = Structural()
# sentiment_features = Sentiment()
# df1 = content_features.extract_features(X)
# df2 = structural_features.extract_features(X)
# df3 = sentiment_features.extract_features(X)
# df = pd.concat([df1, df2], axis=1)
# df = pd.concat([df, df3], axis=1)
# df.to_csv("data/msdialogue/train_data.csv", index=False)

# labels = ['OQ', 'RQ', 'CQ', 'FD', 'FQ', 'IR', 'PA', 'PF', 'NF', 'GG', 'JK', 'O']
# label_dict = dict(list(zip(labels, range(0, len(labels)))))
#
# X, y = load_from_json()
# X = load_from_csv()
# y = preprocess_labels(y, label_dict)
# ml = MLModels()
#
# max_idx = 0
# max_acc = 0
# max_vec = []
# kf = KFold(n_splits=10, random_state=42)
# for idx, train_test_index in enumerate(kf.split(X)):
#     print(train_test_index[0])
#     X_train, X_test = pd.DataFrame(np.array(X)[np.array(train_test_index[0])]), \
#                       pd.DataFrame(np.array(X)[np.array(train_test_index[1])])
#     y_train, y_test = np.array(y)[train_test_index[0]], np.array(y)[train_test_index[1]]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
#     acc, f1 = ml.test_basic_model_multilabel(X_train, y_train, X_test, y_test)
#     if np.sum(np.array(acc)) > max_acc:
#         max_acc = np.sum(np.array(acc))
#         max_vec = acc
#         max_idx = idx
# print("Accuracy: {}".format(max_acc))
# print("Vec:")
# print(max_vec)
# print("Index: {}".format(max_idx))

# from models.ml_models import MLModels
#
# ml = MLModels()
# ml.best_params(X_train, X_test, y_train, y_test)