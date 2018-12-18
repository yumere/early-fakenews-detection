import argparse
import json

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support
from torch.nn import functional as F
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

            tqdm.write("Evaluating on {:,}".format(epoch))
            precision, recall, f1, loss = evaluate(model, dev_loader, config)
            tqdm.write("Label False | Precision: {:6.4f} | Recall: {:6.4f} | F1-Score: {:6.4f} | Loss: {:.7f}".format(precision[0], recall[0], f1[0], loss))
            tqdm.write("Label True  | Precision: {:6.4f} | Recall: {:6.4f} | F1-Score: {:6.4f} | Loss: {:.7f}".format(precision[1], recall[1], f1[1], loss))
            tqdm.write("\n")


def evaluate(model: DetectModel, loader: DataLoader, config: dict)->tuple:
    batch_size = loader.batch_size
    device = torch.device("cuda:{}".format(args.cuda) if args.cuda else 'cpu')
    criterion = nn.CrossEntropyLoss(reduction='none')

    total_loss = []
    total_output = []
    total_labels = []

    with torch.no_grad():
        initial_hidden_state = torch.zeros(config['rnn_layers'], batch_size, config['hidden_size'], dtype=torch.float).to(device)
        for step, (seq_inputs, labels) in enumerate(loader):
            seq_inputs = torch.tensor(seq_inputs, dtype=torch.float).to(device)
            labels = torch.tensor(labels, dtype=torch.long).to(device)

            output = model(seq_inputs, initial_hidden_state)
            loss = criterion(output, labels)
            total_loss.append(loss)
            output = F.softmax(output, dim=1).argmax(dim=1)
            total_output.append(output)
            total_labels.append(labels)

        total_loss = torch.cat(total_loss).contiguous().mean()
        total_output = torch.cat(total_output).cpu().numpy()
        total_labels = torch.cat(total_labels).cpu().numpy()

        precision, recall, f1, _ = precision_recall_fscore_support(total_labels, total_output, labels=[0, 1])

    return precision, recall, f1, total_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train')
    parser.add_argument('--config', type=str, help="config file path (required)", required=True)
    parser.add_argument('--topic_feature', action='store_true', default=False, help="Use topic feature of tweet")
    parser.add_argument('--train_file', type=str, help="data file for train", required=True)
    parser.add_argument('--dev_file', type=str, help="data file for dev")

    parser.add_argument('--logdir', type=str, help="log directory (optional)")
    parser.add_argument('--lr', type=float, default=0.2, metavar="0.2", help="learning rate for model")
    parser.add_argument('--batch_size', type=int, default=32, metavar='32', help="batch size for learning")
    parser.add_argument('--epoch', type=int, default=10, metavar="10", help="the number of epochs")
    parser.add_argument('-n', '--num_workers', type=int, default=0, help="CPUs for dataloader")
    parser.add_argument('--cuda', type=int, default=None, help="GPU number (default: None=CPU)")

    args = parser.parse_args()
    model_config = json.load(open(args.config, "rt"))['model']

    if args.mode == 'train':
        train(args, model_config)

    elif args.mode == 'test':
        pass
