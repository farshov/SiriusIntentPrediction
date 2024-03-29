import nltk
import pandas as pd
from nltk.stem import PorterStemmer
from feature_extraction.feature_extractor import FeatureExtractor
from collections import Counter

def tokenizer(utter):
    pattern = re.compile(r"(?u)\b\w\w+\b")
    return pattern.findall(utter)

class Structural(FeatureExtractor):
    def __init__(self):
        super().__init__()
        self.stops = nltk.corpus.stopwords.words('english')
        self.stemmer = PorterStemmer()
        self.tf_idf_dict = {}

    def get_cosine_similarity_two_utters(self, first_utter, second_utter):
        first_term_freq = Counter(first_utter)
        second_term_freq = Counter(second_utter)
        first_norm = 0.0
        second_norm = 0.0
        dot_product = 0.0
        for word, count in first_term_freq.items():
            idf = self.tf_idf_dict[word]
            first_norm += pow(count * idf, 2)
            if word in second_term_freq.keys():
                dot_product += count * second_term_freq[word] * pow(idf, 2)
        for word, count in second_term_freq.items():
            idf = self.tf_idf_dict[word]
            second_norm += pow(count * idf, 2)
        similarity = float(dot_product) / (pow(first_norm, 0.5) * pow(second_norm, 0.5))


    def remove_stop_words(self, utterance):
        """
        Removes stop words from utterance.
        :param utterance: string with utterance
        :return: list of words
        """
        tokens = nltk.word_tokenize(utterance)
        return [token for token in tokens if token not in self.stops]

    def remove_stop_words_stem(self, utterance):
        """
        Removes stop words from stemmed utterance.
        :param utterance: string with utterance
        :return: list of words
        """
        tokens = nltk.word_tokenize(utterance)
        tokens = [self.stemmer.stem(token) for token in tokens]
        return self.remove_stop_words(" ".join(tokens))

    def extract_features(self, data):
        """
        Extracts features from data.
        :param data: list of lists of tuple(sender, strs), containing dialogues with utterances
        :return: pandas DataFrame with features
        """
        abs_feature = []
        norm_feature = []
        utter_len_feature = []
        utter_len_un_feature = []
        utter_len_stem_un_feature = []
        is_start_feature = []

        for dialogue in data:
            abs_feature += [(i+1) for i in range(len(dialogue))]
            frac = 1/len(dialogue)
            norm_feature += [i*frac for i in range(len(dialogue))]

            utter_len = []
            is_starter = []
            uniq_utter = []
            stem_uniq_utter = []

            for utterance in dialogue:
                starter = utterance[0]
                utter = utterance[1].translate(self.table).lower()
                rem_stops = self.remove_stop_words(utter)
                rem_stops_stem = self.remove_stop_words_stem(utter)
                utter_len.append(len(rem_stops))
                uniq_utter.append(len(set(rem_stops)))
                stem_uniq_utter.append(len(set(rem_stops_stem)))
                is_starter.append(1 if starter == dialogue[0][0] else 0)

            utter_len_feature += utter_len
            utter_len_un_feature += uniq_utter
            utter_len_stem_un_feature += stem_uniq_utter
            is_start_feature += is_starter

        ans_df = pd.DataFrame()
        ans_df["absolute_position"] = abs_feature
        ans_df["normalized_position"] = norm_feature
        ans_df["utterance_length"] = utter_len_feature
        ans_df["utterance_length_unique"] = utter_len_un_feature
        ans_df["utterance_length_stemmed_unique"] = utter_len_stem_un_feature
        ans_df["is_starter"] = is_start_feature

        return ans_df
