import pandas as pd
from data_load.loading import preprocess_labels
from models.ml_models import MLModels
import numpy as np
from quality_metrics.stat_tests import perform_stat_tests
import warnings
warnings.filterwarnings("ignore")

labels = ['OQ', 'RQ', 'CQ', 'FD', 'FQ', 'IR', 'PA', 'PF', 'NF', 'GG', 'JK', 'O']
label_dict = dict(list(zip(labels, range(0, len(labels)))))

ml = MLModels()
models = ["KNN", "NaiveBayes", "SVM", "RandomForest", "AdaBoost"]

df = pd.read_csv("data/msdialogue/train_features.tsv", header=None, delimiter="\t")
arr = df.iloc[:, 1]
X_train_theirs = pd.DataFrame(np.array(list(map(lambda x: x.split(), arr))).astype("float32"))
y_train = preprocess_labels(list(df.iloc[:, 0]), label_dict)

df = pd.read_csv("data/msdialogue/test_features.tsv", header=None, delimiter="\t")
arr = df.iloc[:, 1]
X_test_theirs = pd.DataFrame(np.array(list(map(lambda x: x.split(), arr))).astype("float32"))
y_test = preprocess_labels(list(df.iloc[:, 0]), label_dict)

df = pd.read_csv("data/msdialogue/token_train.csv")
X_train_our = df
df = pd.read_csv("data/msdialogue/token_test.csv")
X_test_our = df

predictions_theirs = ml.get_basic_model_predictions(X_train_theirs, y_train, X_test_theirs)
predictions_our = ml.get_basic_model_predictions(X_train_our, y_train, X_test_our)

for i, model in enumerate(models):
    print("MODEL: {}".format(model))
    perform_stat_tests(predictions_our[i], predictions_theirs[i], y_test)
    print("\n")
