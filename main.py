import argparse
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import DetectModel
from utils.data import SeqDataset, collate_fn


def predict(args, config):
    pass


def train(args: argparse.Namespace, config: dict):
    data = [json.loads(d) for d in open(args.input, "rt").readlines()]
    dataset = SeqDataset(data)

    device = torch.device("cuda:{}".format(args.cuda) if args.cuda else "cpu")

    model = DetectModel(input_size=config['model']['input_size'], hidden_size=config['model']['hidden_size'], rnn_layers=config['model']['rnn_layers'],
                        out_channels=config['model']['out_channels'], height=config['model']['height'], cnn_layers=config['model']['cnn_layers'],
                        linear_hidden_size=config['model']['linear_hidden_size'], linear_layers=config['model']['linear_layers'], output_size=config['model']['output_size'])
    model = model.to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
    h0 = torch.zeros(config['model']['rnn_layers'], args.batch_size, config['model']['hidden_size']).to(device)

    for epoch in tqdm(range(args.epoch), desc="Epochs"):
        with tqdm(total=len(dataset), desc="Sequences", leave=False) as pbar:
            for step, (sequences, labels) in enumerate(dataloader):
                pbar.update(args.batch_size)

                model.zero_grad()
                h0.data.zero_()

                sequences = torch.tensor(sequences, dtype=torch.float, requires_grad=False).to(device)
                labels = torch.tensor(labels, dtype=torch.long, requires_grad=False).to(device)

                output = model(sequences, h0)

                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                if step % 100 == 0:
                    tqdm.write("Step: {:,} Loss: {:,}".format(step, loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="config file path", required=True)
    parser.add_argument('--cuda', type=int, default=None, help="GPU number (default: None=CPU)")
    parser.add_argument('--input', type=str, help="input file path", required=True)
    parser.add_argument('--learning-rate', type=float, default=0.2, metavar="0.2", help="learning rate for model")
    parser.add_argument('--batch-size', type=int, default=32, metavar='32', help="batch size for learning")
    parser.add_argument('--epoch', type=int, default=10, metavar="10", help="the number of epochs")

    args = parser.parse_args()

    config_json = json.load(open(args.config, "rt"))

    train(args, config_json)
