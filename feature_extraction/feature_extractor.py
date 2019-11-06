import string


class FeatureExtractor:
    def __init__(self):
        self.table = str.maketrans('', '', string.punctuation)

    def extract_features(self, data):
        """
        Extracts features from data.
        :param data: list of lists of tuple(sender, strs), containing dialogues with utterances
        :return: pandas DataFrame with features
        """
        pass

