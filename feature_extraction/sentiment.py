import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import opinion_lexicon
from feature_extraction.feature_extractor import FeatureExtractor


class Sentiment(FeatureExtractor):
    """
    Extractor of sentimental features of a dataset.
    """

    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.negative_lexicon = opinion_lexicon.negative()
        self.positive_lexicon = opinion_lexicon.positive()

    def utterance_contains_thank(self, utterance: str) -> bool:
        """

        :param utterance: utterance from a dialogue
        :return: Does the utterance contain the keyword thank
        """
        return "thank" in utterance

    def utterance_contains_excl_mark(self, utterance: str) -> bool:
        """

        :param utterance: utterance from a dialogue
        :return: Does the utterance contain an exclamation mark
        """
        return '!' in utterance

    def utterance_is_feedback(self, utterance) -> bool:
        """

        :param utterance: utterance from a dialogue
        :return: Does the utterance contain the keyword did not, does not
        """
        return "does not" in utterance or "did not" in utterance

    def utterance_compute_sentiment_scores(self, utterance) -> [float]:
        """

        :param utterance: utterance from a dialogue
        :return:  Sentiment scores of the utterance computed by VADER [8] (negative, neutral, and positive)
        """
        scores = self.sentiment_analyzer.polarity_scores(utterance)
        return [scores["neg"], scores["neu"], scores["pos"]]

    def utterance_count_opinion_lexicon(self, utterance) -> [int]:
        """

        :param utterance: utterance from a dialogue
        :return:  Number of positive and negative words from an opinion lexicon
        """
        num_positive_words = 0
        num_negative_words = 0
        for word in utterance:
            if word in self.negative_lexicon:
                num_negative_words += 1
            elif word in self.positive_lexicon:
                num_positive_words += 1
        return [num_positive_words, num_negative_words]

    def extract_features(self, data, normalized_data):

        """

        :param data: list of lists, containing dialogues with utterances
        :return: pandas DataFrame with sentimental features respecting to the utterance id:
                   | thank | exclamation_mark | feedback | sentiment_scores | opinion_lexicon |
        """

        df = pd.DataFrame({'thank': [], 'exclamation_mark': [], 'feedback': [],
                           'sentiment_neg': [], 'sentiment_neu': [], 'sentiment_pos': [], 'num_positive_lexicon': [],
                           'num_negative_lexicon': []})

        for dialogue in normalized_data:
            for utterance in dialogue:
                thank = self.utterance_contains_thank(utterance)
                exclamation_mark = self.utterance_contains_excl_mark(utterance)
                feedback = self.utterance_is_feedback(utterance)
                sentiment_neg, sentiment_neu, sentiment_pos = self.utterance_compute_sentiment_scores(utterance)
                num_positive_lexicon, num_negative_lexicon = self.utterance_count_opinion_lexicon(utterance)
                utterance_info = pd.DataFrame({'thank': [thank], 'exclamation_mark': [exclamation_mark],
                                               'feedback': [feedback], 'sentiment_neg': [sentiment_neg],
                                               'sentiment_neu': [sentiment_neu], 'sentiment_pos': [sentiment_pos],
                                               'num_positive_lexicon': [num_positive_lexicon],
                                               'num_negative_lexicon': [num_negative_lexicon]})
                df.append(utterance_info, ignore_index=True)

        return df

