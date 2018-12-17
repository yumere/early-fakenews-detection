import numpy as np
from torch.utils.data import Dataset


class SeqDataset(Dataset):
    def __init__(self, dataset):
        super(SeqDataset, self).__init__()
        self.dataset = dataset

    def __getitem__(self, item):
        seq_data, text, label, user_id, created_at = self.dataset[item]
        return seq_data, label

    def __len__(self):
        return len(self.dataset)

    @property
    def seq_len(self):
        return len(self.dataset[0][0])

    @property
    def input_size(self):
        return len(self.dataset[0][0][0])

    @staticmethod
    def collate_fn(data):
        seq_data, labels = zip(*data)

        return seq_data, labels


def dataset_test(file_path):
    from torch.utils.data import DataLoader
    import ujson as json
    with open(file_path, "rt", encoding="utf-8") as f:
        data = [json.loads(d) for d in f]

    dataset = SeqDataset(data)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True, num_workers=0, collate_fn=SeqDataset.collate_fn)

    for i, j in enumerate(loader):
        if i == 100:
            break

        print(j)


if __name__ == '__main__':
    dataset_test(file_path="train_5.json")
