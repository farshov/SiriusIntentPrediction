import pandas as pd
import nltk.data
from gensim.models import Word2Vec
import string
import random
from tqdm import tqdm

MAX_SEQ_LEN = 150


def train_word2vec(path='data/msdialogue/MSDialog-Complete.json', save_path='PreTrainedWord2Vec',
                   size=100, window=5, min_count=3, epochs=30, seed=42):
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
    print('Building corpus')
    data = pd.read_json(path, orient='index')

    text = []
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    # exclude = set(string.punctuation)
    random.seed(seed)
    for i in tqdm(range(len(data))):
        dialog = data.iloc[i]['utterances']
        for j in range(len(dialog)):
            sentences = tokenizer.tokenize(dialog[j]['utterance'].lower())
            for sent in sentences:
                sent = ''.join(sent).split()  # ch for ch in sent if ch not in exclude and not ch.isdigit()).split()
                text.append(sent)
    print('Training Word2Vec')
    model = Word2Vec(text, size=size, window=window, min_count=min_count, iter=epochs, seed=seed, workers=-1)
    model.wv.save_word2vec_format(save_path)
