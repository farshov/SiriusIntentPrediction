from torch.utils import data


class MSDialog(data.Dataset):
    def __init__(self, data, labels):
        self.X = data
        self.y = labels

    def __len__(self):
        return data.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]