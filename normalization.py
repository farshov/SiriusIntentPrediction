import nltk
from nltk.stem.snowball import SnowballStemmer


def normilize(convs):
    """

    :param convs: list of lists of tuple(sender, message)
    :return: list of lists of tuple(sender, norm_message)
    """
    stops = nltk.corpus.stopwords.words('english')
    stemmer = SnowballStemmer("english")

    convs_norm = []
    for conv in convs:

        clean_conv = []
        for utterance in conv:
            starter = utterance[0]
            utter = utterance[1]
            tokens = nltk.word_tokenize(utter)
            rem_stops = ' '.join([stemmer.stem(token) for token in tokens if not token in stops])
            clean_conv.append((starter, rem_stops))
        convs_norm.append(clean_conv)

    return convs_norm
