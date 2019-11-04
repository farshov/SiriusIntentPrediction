import nltk
import numpy as np
from nltk.stem.snowball import SnowballStemmer
import re

class Structural:
    def structural_features(self, convs, normalised_convs):
        stops = nltk.corpus.stopwords.words('english')
        stemmer = SnowballStemmer("english")

        abs_feature = []
        norm_feature = []
        utter_len_feature = []
        utter_len_un_feature = []
        utter_len_stem_un_feature = []
        is_start_feature = []

        for (conv, norm_conv) in zip(convs, normalised_convs):
            abs_feature += [i for i in range(len(conv))]
            frac = 1/len(conv)
            norm_feature += [i*frac for i in range(len(conv))]

            utter_len = []
            is_starter = []
            uniq_utter = []
            stem_uniq_utter = []

            for utterance in conv:
                starter = utterance[0]
                utter = utterance[1]
                tokens = nltk.word_tokenize(utter)
                rem_stops = [token for token in tokens if not token in stops]
                utter_len.append(len(tokens))
                uniq_utter.append(len(rem_stops))
                is_starter.append(starter == conv[0][0])

            for utterance in norm_conv:
                stem_uniq_utter.append(len(list(np.unique(nltk.word_tokenize(utterance[1])))))

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
