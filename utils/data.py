import json
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class SeqDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        super(SeqDataset, self).__init__()

    def __getitem__(self, item):
        sequence, label = self.dataset[item]
        return sequence, label

    def __len__(self):
        return len(self.dataset)

    @property
    def seq_len(self):
        return len(self.dataset[0][0])

    @property
    def input_size(self):
        return len(self.dataset[0][0][0])


def collate_fn(data):
    sequences, labels = zip(*data)

    return sequences, labels


if __name__ == '__main__':
    """
    Example usage of dataset and dataloader
    """

    file = "../output_20.dat"
    data = [json.loads(d) for d in open(file, "rt").readlines()]

    dataset = SeqDataset(data)
    for i, (sequences, labels) in enumerate(DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn, drop_last=True)):
        if i == 10:
            break
        print(torch.tensor(sequences, dtype=torch.float).shape)
        print(torch.tensor(labels, dtype=torch.long).shape)
        print()
