import pandas as pd
from sklearn.model_selection import train_test_split
from data_load.loading import load_from_json
from data_load.loading import preprocess_labels
from feature_extraction.content_feature_extractor import Content
from feature_extraction.structural_feature_extractor import Structural
from models.ml_models import MLModels

labels = ['OQ', 'RQ', 'CQ', 'FD', 'FQ', 'IR', 'PA', 'PF', 'NF', 'GG', 'JK', 'O']
label_dict = dict(list(zip(labels, range(0, len(labels)))))

X, y = load_from_json()

content_features = Content()
structural_features = Structural()
df1 = content_features.extract_features(X)
df2 = structural_features.extract_features(X)
df = pd.concat([df1, df2], axis=1)

X = df
y = preprocess_labels(y, label_dict)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.125, random_state=42)

ml = MLModels()
ml.test_basic_model_multilabel(X_train, y_train, X_test, y_test)
