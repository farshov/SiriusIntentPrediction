import pandas as pd
from collections import Counter
import random

def get_convs(path):
    df = pd.read_csv(path)
    dialog_ids = df["dialogueID"].unique()
    starters = {}
    for i in range(len(df)):
        if not df["dialogueID"][i] in starters:
            starters[df["dialogueID"][i]] = df["from"][i]
    convs = [0] * len(dialog_ids)
    i = 0
    for dialog_id in dialog_ids:
        convs[i] = list(zip(df[df["dialogueID"] == dialog_id]["from"].tolist(),
                    df[df["dialogueID"] == dialog_id]["text"].tolist()))
        i += 1

    return convs


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
