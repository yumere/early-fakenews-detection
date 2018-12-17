import argparse
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import SeqDataset
from fakenews_detection import DetectModel


def train(args, config: dict):
    device = torch.device("cuda:{}".format(args.cuda) if args.cuda else 'cpu')
    with open(args.train_file, "rt", encoding="utf-8") as train_f:
        train_dataset = [json.loads(d) for d in train_f]
        train_dataset = SeqDataset(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers, collate_fn=SeqDataset.collate_fn)

    if args.dev_file:
        with open(args.dev_file, "rt", encoding="utf-8") as dev_f:
            dev_dataset = [json.loads(d) for d in dev_f]
            dev_dataset = SeqDataset(dev_dataset)
            dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers, collate_fn=SeqDataset.collate_fn)

    model = DetectModel(config['input_size'],
                        config['hidden_size'], config['rnn_layers'],
                        config['out_channels'], config['height'], config['cnn_layers'],
                        config['linear_hidden_size'], config['linear_layers'], config['output_size'])
    model = model.to(device)

    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    initial_hidden_state = torch.zeros(config['rnn_layers'], args.batch_size, config['hidden_size'], dtype=torch.float, requires_grad=False).to(device)
    for epoch in range(args.epoch):

        lr = args.lr * (0.9) ** epoch
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with tqdm(total=len(train_dataset), desc="Train", ncols=75) as pbar:
            for step, (seq_inputs, labels) in enumerate(train_loader):
                pbar.update(len(seq_inputs))
                optimizer.zero_grad()
                seq_inputs = torch.tensor(seq_inputs, dtype=torch.float, requires_grad=False).to(device)
                labels = torch.tensor(labels, dtype=torch.long, requires_grad=False).to(device)

                output = model(seq_inputs, initial_hidden_state[:, :seq_inputs.shape[0], :])
                loss = criterion(output, labels)
                loss.backward()

                optimizer.step()

                if step % 500 == 0:
                    tqdm.write("Epoch: {:3,} / Step: {:5,} / Loss: {:10.5f} / Learning rate: {:.4f}".format(epoch, step, loss.item(), lr))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="config file path (required)", required=True)
    parser.add_argument('--cuda', type=int, default=None, help="GPU number (default: None=CPU)")
    parser.add_argument('--logdir', type=str, help="log directory (optional)")
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train')

    parser.add_argument('--train_file', type=str, help="data file for train", required=True)
    parser.add_argument('--dev_file', type=str, help="data file for dev")
    parser.add_argument('--lr', type=float, default=0.2, metavar="0.2", help="learning rate for model")
    parser.add_argument('--batch_size', type=int, default=32, metavar='32', help="batch size for learning")
    parser.add_argument('--epoch', type=int, default=10, metavar="10", help="the number of epochs")
    parser.add_argument('-n', '--num_workers', type=int, default=0, help="CPUs for dataloader")

    args = parser.parse_args()
    model_config = json.load(open(args.config, "rt"))['model']

    if args.mode == 'train':
        train(args, model_config)

    elif args.mode == 'test':
        pass
