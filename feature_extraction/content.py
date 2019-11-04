import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Content:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def fit_transform(self, dialogues):
        """
        :param dialogues: list of lists of strs
        :return: pd.DataFrame
        """
        self.vectorizer.fit(np.array(dialogues).flatten())
        return self.transform(dialogues)

    def transform(self, dialogues):
        """
        :param dialogues: list of lists of strs
        :return: pd.DataFrame
        """
        features = []
        for i, dialog in enumerate(dialogues):
            dialog = self.vectorizer.transform(dialog)
            for j in range(dialog.shape[0]):
                similarity = cosine_similarity(dialog[0], dialog[j])[0][0]
                features.append([similarity] + self.get5w1h(dialogues[i][j]))

        features = pd.DataFrame(features, columns=['cos_dist', 'what', 'where', 'how',
                                                   'when', 'why', 'who'])
        return features

    def get5w1h(self, phrase):
        """
        :param phrase: str
        :return: array with labels of interrogative words
        """
        mapping = {'what': 0,
                   'where': 1,
                   'how': 2,
                   'when': 3,
                   'why': 4,
                   'who': 5}

        labels = [0] * len(mapping)
        phrase = set(phrase.lower().split())
        for word in mapping:
            if word in phrase:
                labels[mapping[word]] = 1
        return labels
