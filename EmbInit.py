import pandas as pd
import nltk.data
from gensim.models import Word2Vec
import string
import nltk.data
import random
from tqdm import tqdm


def train_word2vec(path='data/msdialogue/MSDialog-Complete.json', save_path='PreTrainedWord2Vec',
                   size=100, window=5, min_count=10, iter=20, seed=42):
    """

    :param path: Dataset for training an embedding
    :param save_path: path to file to save
    :param size: dimentionality of emvedding
    :param window: gensim Word2Vec param
    :param min_count: gensim Word2Vec param
    :param iter: gensim Word2Vec param
    :param seed: random seed
    :return: None
    """
    data = pd.read_json(path, orient='index')

    text = []
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    exclude = set(string.punctuation)
    random.seed(seed)
    for i in tqdm(range(len(data))):
        dialog = data.iloc[i]['utterances']
        for j in range(len(dialog)):
            sentences = tokenizer.tokenize(dialog[j]['utterance'].lower())
            for sent in sentences:
                sent = ''.join(ch for ch in sent if ch not in exclude and not ch.isdigit()).split()
                if sent and not i % 25:
                    sent[random.randint(0, len(sent) - 1)] = 'UNK'
                text.append(sent)

    model = Word2Vec(text, size=size, window=window, min_count=min_count, iter=iter, seed=seed, workers=-1)
    model.wv.save_word2vec_format(save_path)

    return None
