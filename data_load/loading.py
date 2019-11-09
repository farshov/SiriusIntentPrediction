import pandas as pd
from collections import Counter
import random
num_tags = 12


def get_labels(data):
    """
    :param data: pandas DataFrame which contains labels in column 'tags', appropriate messages in column 'utterances'
    :return: set of 33 most common labels for our model
    """
    useless = {'JK', 'GG', 'O'}
    combinations = []
    for i in range(len(data)):
        d = data.iloc[i]['utterances']
        for utt in d:
            comb = set(utt['tags'].split())
            if len(comb) > 1:
                inter = comb.intersection(useless)
                if len(inter) < len(comb):
                    for to_del in inter:
                        comb.remove(to_del)
                comb = sorted(list(comb))
            combinations.append('_'.join(comb))

    labels = Counter(combinations).most_common(32)
    labels = [label[0] for label in labels] + ['O']
    return set(labels)


def generate_dataset(data, labels):
    """
    :param data: pandas DataFrame which contains labels in column 'tags', appropriate messages in column 'utterances'
    :param labels: set of 33 most common labels for our model
    :return: X - list of lists of tuple(sender, str with message),
            y - list of strs which are label of appropriate message
    """
    dialogs = []
    targets = []
    labels_set = set(labels)
    for i in range(len(data)):
        d = data.iloc[i]['utterances']
        diag = []
        for utt in d:
            diag.append((utt['user_id'], utt['utterance']))
            comb = sorted(utt['tags'].split())
            if 'GG' in comb and len(comb) > 1:
                comb.remove('GG')
            if '_'.join(comb) in labels_set:
                targets.append('_'.join(comb))
            else:
                num = random.randint(0, len(comb) - 1)
                targets.append(comb[num])
        dialogs.append(diag)

    return dialogs, targets


def load_from_json(path='data/msdialogue/MSDialog-Intent.json', seed=42):
    """
    :param path: path to data file
    :param seed: random seed for transforming rare labels
    :return: X - list of lists of tuple(sender, str with message),
            y - list of strs which are label of appropriate message
    """
    data = pd.read_json(path, orient='index')

    # getting useful combinations
    random.seed(seed)
    labels = get_labels(data)

    X, y = generate_dataset(data, labels)

    return X, y


def load_from_csv(path='data/msdialogue/out.csv'):
    return pd.read_csv(path)


def encode_labels(all_labels, label_dict):

    """
    params:
    string labels for encoding
    dict {(label, index)}
    
    return: indices
    """

    res = []
    for labels in all_labels:
        encode = labels.split('_')
        i = 0
        for label in encode:
            encode[i] = label_dict[label] 
            i += 1
        res += [encode]
    return res


def preprocess_labels(labels, labels_dict):

    """
    params:
    list of list of string labels
    dict {(label, index)}
    return:
    vector where i-th element is 1 when i-th list contains element i
    """

    res = []
    encode = encode_labels(labels, labels_dict)
    for line in encode:
        vector = [0]*num_tags
        for num in line:
            vector[num] = 1
        res.append(vector)
    return res
