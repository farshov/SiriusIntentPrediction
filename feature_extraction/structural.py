import nltk
import numpy as np
from nltk.stem.snowball import SnowballStemmer
import re


def tokenize(utterance):
    pattern = re.compile(r"(?u)\b\w\w+\b")
    return pattern.findall(utterance)


def structural_features(convs):
    stops = nltk.corpus.stopwords.words('english')
    stemmer = SnowballStemmer("english")
    
    abs_feature = []
    norm_feature = []
    ut_len_feature = []
    ut_len_un_feature = []
    ut_len_stem_un_feature = []
    is_start_feature = []
    
    for conv in convs:
        abs_feature += [i for i in range(len(conv))]
        norm_feature += [i/len(conv) for i in range(len(conv))]
        
        ut_len = []
        is_starter = []
        uniq_ut = []
        stem_uniq_ut = []
        
        for utterance in conv:
            starter = utterance[0]
            ut = utterance[1]
            tokens = tokenize(ut)
            rem_stops = [token for token in tokens if not token in stops]
            ut_len.append(len(tokens))
            uniq_ut.append(len(rem_stops))
            stem_uniq_ut.append(len(list(np.unique([stemmer.stem(token) for token in rem_stops]))))
            is_starter.append(starter == conv[0][0])
        
        ut_len_feature += ut_len
        ut_len_un_feature += uniq_ut
        ut_len_stem_un_feature += stem_uniq_ut
        is_start_feature += is_starter
        
    ans_df = pd.DataFrame()
    ans_df["absolute_position"] = abs_feature
    ans_df["normalized_position"] = norm_feature
    ans_df["utterance_length"] = ut_len_feature
    ans_df["utterance_length_unique"] = ut_len_un_feature
    ans_df["utterance_length_stemmed_unique"] = ut_len_stem_un_feature
    ans_df["is_starter"] = is_start_feature
    
    return ans_df
