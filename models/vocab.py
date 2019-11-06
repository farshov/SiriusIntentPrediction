import torch


class Vocab:
    def __init__(self):
        self.word2index = {"PAD": 0, "UNK": 1}
        self.index2word = {0: "PAD", 1: "UNK"}
        self.n_words = 2

    def build(self, corpus):
        """
        :param corpus: list of strs containing sentenses
        :return: None
        """
        for sent in corpus:
            self.add_sentence(sent)

    def add_sentence(self, sentence):
        """
        :param sentence: str contains sentense
        :return: None
        """
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        """
        :param word: str contains word
        :return:
        """
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

    def __len__(self):
        return self.n_words

    def tokenize(self, corpus, max_len):
        """
        :param corpus: list of strs containing sentenses
        :return:
        """
        tok_corp = []
        tok_corp.append(torch.zeros(max_len))
        for sent in corpus:
            tok_sent = []
            for word in sent.split(' '):
                tok_sent.append(self.word2index.get(word, self.word2index["UNK"]))
            tok_corp.append(torch.tensor(tok_sent))
        return tok_corp

    def get_pad(self):
        return self.word2index['PAD']
