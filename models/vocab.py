import torch
import gensim


class Vocab:
    def __init__(self):
        self.word2index = dict()  # {"PAD": 0, "UNK": 1}
        self.index2word = dict()  # {0: "PAD", 1: "UNK"}
        self.n_words = 0

    def build(self, embedder):
        """
        :param embedder: gensim pretrained Word2Vec
        :return:
        """
        words = embedder.index2word

        for i in range(len(words)):
            self.word2index[words[i]] = i
            self.index2word[i] = words[i]
            self.n_words += 1
        # self.word2index['PAD'] = self.n_words
        # self.index2word[self.n_words] = 'PAD'
        # self.n_words += 1

    def __len__(self):
        return self.n_words

    def tokenize(self, corpus, max_len):
        """
        :param corpus: list of strs containing sentenses
        :return:
        """
        tok_corp = list()
        tok_corp.append(torch.zeros(max_len))
        for sent in corpus:
            tok_sent = []
            for word in sent.split(' '):
                tok_sent.append(self.word2index.get(word, self.word2index["<UNK>"]))
            tok_corp.append(torch.tensor(tok_sent))
        return tok_corp

    def get_pad(self):
        return self.word2index['<PAD>']
