import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import opinion_lexicon
from feature_extraction.feature_extractor import FeatureExtractor
from threading import Thread
import multiprocessing
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.downloader.download('opinion_lexicon')


class Sentiment(FeatureExtractor):
    """
    Extractor of sentimental features of a dataset.
    """

    def __init__(self):
        super().__init__()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.negative_lexicon = opinion_lexicon.negative()
        self.positive_lexicon = opinion_lexicon.positive()

    @staticmethod
    def utterance_contains_thank(utterance):
        """

        :param utterance: utterance from a dialogue
        :return: Does the utterance contain the keyword thank
        """
        return 1 if "thank" in utterance else 0

    @staticmethod
    def utterance_contains_excl_mark(utterance):
        """

        :param utterance: utterance from a dialogue
        :return: Does the utterance contain an exclamation mark
        """
        return 1 if '!' in utterance else 0

    @staticmethod
    def utterance_is_feedback(utterance):
        """

        :param utterance: utterance from a dialogue
        :return: Does the utterance contain the keyword "did not", "does not"
        """
        return 1 if "does not" in utterance or "did not" in utterance \
               or "doesnt" in utterance or "didnt" in utterance else 0

    def utterance_compute_sentiment_scores(self, utterance):
        """

        :param utterance: utterance from a dialogue
        :return: Sentiment scores of the utterance computed by VADER
        [C. J. Hutto and E. Gilbert. VADER: A Parsimonious Rule-Based Model
         for Sentiment Analysis of Social Media Text. In ICWSM ’14, 2014]
          (negative, neutral, and positive)
        """
        scores = self.sentiment_analyzer.polarity_scores(" ".join(utterance))
        return [scores["neg"], scores["neu"], scores["pos"]]

    def utterance_count_opinion_lexicon(self, utterance):
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

    def get_utterance_info(self, df, idx, sender_utterance):
        utterance = sender_utterance[1].split()
        preprocessed_utterance = sender_utterance[1].translate(self.table).lower().split()
        thank = self.utterance_contains_thank(preprocessed_utterance)
        exclamation_mark = self.utterance_contains_excl_mark(utterance)
        feedback = self.utterance_is_feedback(preprocessed_utterance)
        sentiment_neg, sentiment_neu, sentiment_pos = self.utterance_compute_sentiment_scores(preprocessed_utterance)
        num_positive_lexicon, num_negative_lexicon = self.utterance_count_opinion_lexicon(preprocessed_utterance)
        df.loc[idx, :] = [thank, exclamation_mark, feedback, sentiment_neg, sentiment_neu, sentiment_pos,
                          num_positive_lexicon, num_negative_lexicon]

    def extract_features(self, data):
        """

        :param data: list of lists of tuple(sender, strs), containing dialogues with utterances
        :return: pandas DataFrame with sentimental features respecting to the utterance id:
                   | thank | exclamation_mark | feedback | sentiment_scores | opinion_lexicon |
        """
        data = [item for sublist in data for item in sublist]
        data = list(enumerate(data))
        num_utter = len(data)
        df = pd.DataFrame({'thank': [0]*num_utter, 'exclamation_mark': [0]*num_utter, 'feedback': [0]*num_utter,
                           'sentiment_neg': [0]*num_utter, 'sentiment_neu': [0]*num_utter,
                           'sentiment_pos': [0]*num_utter, 'num_positive_lexicon': [0]*num_utter,
                           'num_negative_lexicon': [0]*num_utter})

        max_threads = multiprocessing.cpu_count()
        threads = []
        for idx, sender_utterance in data:
            process = Thread(target=self.get_utterance_info, args=[df, idx, sender_utterance])
            threads.append(process)
            if len(threads) == max_threads:
                for process in threads:
                    process.join()
                threads = []

        return df

