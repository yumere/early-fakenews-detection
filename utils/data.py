import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from fakenews_detection.topic_feature import TopicFeature


class SeqDataset(Dataset):
    def __init__(self, dataset, topic_feature: TopicFeature):
        self.dataset = dataset
        self.topic_feature = topic_feature
        super(SeqDataset, self).__init__()

    def __getitem__(self, item):
        sequence, tweet, label = self.dataset[item]
        tweet = self.topic_feature.get_feature(tweets=[tweet])
        return sequence, tweet, label

    def __len__(self):
        return len(self.dataset)

    @property
    def seq_len(self):
        return len(self.dataset[0][0])

    @property
    def input_size(self):
        return len(self.dataset[0][0][0])


def collate_fn(data):
    sequences, tweets, labels = zip(*data)

    return sequences, np.concatenate(tweets, axis=0), labels


if __name__ == '__main__':
    """
    Example usage of dataset and dataloader
    """

    file = "../10_train.json"
    data = [json.loads(d) for d in open(file, "rt").readlines()]

    topic_feature = TopicFeature(labels=(1, 0))
    topic_feature.load("../fakenews_detection/data")

    dataset = SeqDataset(data, topic_feature)
    for i, (sequences, tweets, labels) in enumerate(DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn, drop_last=True)):
        if i == 10:
            break
        print(torch.tensor(sequences, dtype=torch.float).shape)
        print(torch.tensor(labels, dtype=torch.long).shape)
        print()
