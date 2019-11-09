import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from feature_extraction.feature_extractor import FeatureExtractor


class Content(FeatureExtractor):
    def __init__(self):
        super().__init__()
        self.vectorizer = TfidfVectorizer()

    def extract_features(self, data):
        """
        Extracts features from data.
        :param data: list of lists, containing dialogues with utterances
        :return: pandas DataFrame with features
        """
        dialogues_tr = []
        for dialog in data:
            dialogues_tr += [phrase[1].translate(self.table).lower() for phrase in dialog]
        self.vectorizer.fit(dialogues_tr)
        return self.transform(data)

    def transform(self, data):
        """
        :param dialogues: list of lists of tuple(sender, strs)
        :return: pd.DataFrame
        """
        dialogues = [[phrase[1].translate(self.table).lower() for phrase in dialog] for dialog in data]
        features = []
        for i, dialog in enumerate(dialogues):
            dialogue = self.vectorizer.transform(dialog).toarray()
            for j in range(dialogue.shape[0]):
                similarity = cosine_similarity([dialogue[0]], [dialogue[j]])[0][0]

                dialog_similarity = cosine_similarity([np.sum(np.delete(dialogue, j, 0), axis=0)], [dialogue[j]])[0][0]
                features.append([similarity] + [dialog_similarity] + [self.contains_question_mark(data[i][j])]
                                + [self.duplicate(dialogues[i][j])] + self.get5w1h(dialogues[i][j]))

        features = pd.DataFrame(features, columns=['initial_utterance_similarity', 'dialog_similarity',
                                                   'question_mark', 'duplicate', 'what', 'where', 'how',
                                                   'when', 'why', 'who'])
        return features

    @staticmethod
    def contains_question_mark(phrase):
        return 1 if '?' in phrase[1] else 0

    def duplicate(self, phrase):
        """
        :param phrase: str
        :return: 1 if "same" or "similar" in phrase 0 otherwise
        """
        duplicate_list = ['same', 'similar']

        phrase = phrase.translate(self.table)
        phrase = set(phrase.lower().split())
        for word in phrase:
            if word in duplicate_list:
                return 1
        return 0

    def get5w1h(self, phrase):
        """
        :param phrase: str
        :return: one-hot-vector indicating 5w1h in phrase
        """
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
        phrase = phrase.translate(self.table)
        phrase = set(phrase.lower().split())
        for word in mapping:
            if word in phrase:
                labels[mapping[word]] = 1
        return labels
