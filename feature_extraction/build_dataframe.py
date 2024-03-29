from feature_extraction.content_feature_extractor import Content
from feature_extraction.structural_feature_extractor import Structural
from feature_extraction.sentiment_feature_extractor import Sentiment
from data_load.loading import load_from_json
import pandas as pd


def build_and_write_dataset(path='data/msdialogue/MSDialog-Intent.json', out_path='out.csv'):
    X, y = load_from_json(path)
    content_features = Content()
    structural_features = Structural()
    sentiment_features = Sentiment()
    df1 = content_features.extract_features(X)
    df2 = structural_features.extract_features(X)
    df3 = sentiment_features.extract_features(X)
    df = pd.concat([df1, df2], axis=1)
    df = pd.concat([df, df3], axis=1)
    df.to_csv(out_path, index=False)
