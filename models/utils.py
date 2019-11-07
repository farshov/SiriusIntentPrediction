import torch


def encode_label(y):
    labels = {'O': 0, 'FQ': 1, 'IR': 2,
              'OQ': 3, 'GG': 4, 'FD': 5,
              'JK': 6, 'NF': 7, 'PF': 8,
              'RQ': 9, 'CQ': 10, 'PA': 11}
    y_encoded = []
    for l in y:
        l = l.split('_')
        cur_y = [0] * len(labels)
        for un_l in l:
            cur_y[labels[un_l]] = 1
        y_encoded.append(cur_y)
    return torch.tensor(y_encoded)
