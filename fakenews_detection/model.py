from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_encoder = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, x: torch.Tensor, h_0: torch.Tensor)->torch.Tensor:
        # x.shape: Batch x sequence length x input_size
        # h_0.shape:
        # output.shape: batch x sequence length x input_size
        output, h_n = self.rnn_encoder(x, h_0)

        # output.shape: batch x input_size
        output = output.mean(dim=1)
        return output


class CNNEncoder(nn.Module):
    def __init__(self, out_channels: int, kernel_size: tuple):
        super(CNNEncoder, self).__init__()
        self.cnn_encoder = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor):
        # x.shape: batch x sequence length x kernel_size[1](input_size)
        # x.shape: batch x 1 x sequence length x kernel_size[1]
        x = x.unsqueeze(dim=1)
        # output.shape: batch x out_channels x sequence length - kernel_size[0] + 1
        output = F.relu(self.cnn_encoder(x))
        # output.shape: batch x out_channels
        output = output.mean(dim=2)
        return output


class DetectModel(nn.Module):
    def __init__(self, input_size,
                 hidden_size, rnn_layers,
                 out_channels, height, cnn_layers,
                 linear_hidden_size, linear_layers, output_size):
        super(DetectModel, self).__init__()
        self.rnn_encoder = RNNEncoder(input_size=input_size, hidden_size=hidden_size, num_layers=rnn_layers)
        self.cnn_encoder = CNNEncoder(out_channels=out_channels, kernel_size=(height, input_size))

        self.linear = nn.Sequential(
            nn.Linear(hidden_size + out_channels, linear_hidden_size), nn.ReLU(inplace=True),
            *chain(*[(nn.Linear(linear_hidden_size, linear_hidden_size), nn.ReLU(inplace=True)) for i in range(linear_layers - 2)]),
            nn.Linear(linear_hidden_size, output_size)
        )

    def forward(self, x, h0):
        # h0 for rnn_encoder
        rnn_output = self.rnn_encoder(x, h0)
        cnn_output = self.cnn_encoder(x)
        cnn_output = cnn_output.squeeze()

        # output.shape: batch x (hidden_size + out_channels)
        # output.shape: batch x output_size
        output = torch.cat([rnn_output, cnn_output], dim=1)
        output = self.linear(output)

        return output


if __name__ == '__main__':
    batch_size = 5
    sequence_length = 10
    input_size = 15
    hidden_size = 5
    out_channels = 10
    height = 3
    linear_hidden_size = 20
    linear_layers = 5
    output_size = 2

    x = torch.randn(batch_size, sequence_length, input_size)
    h0 = torch.zeros(1, batch_size, hidden_size)
    model = DetectModel(input_size, hidden_size, 1, out_channels, height, 1, linear_hidden_size, linear_layers, output_size)
    output = model(x, h0)
    print(output)
