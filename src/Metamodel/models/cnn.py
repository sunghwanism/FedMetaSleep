"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import torch


class LeNet(torch.nn.Module):
    """
    Modern LeNet for inputs of dimension 3x32x32. It adds batch norm layers,
    uses ReLU instead of sigmoids for the activation and replaces average
    pooling with max pooling layers.
    """
    def __init__(self, output_dim):
        super().__init__()
        self.input_dim = 2
        self.output_dim = output_dim

        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=self.input_dim, out_channels=64, kernel_size=7, padding=3),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2),
            torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=7, padding=3),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2),
            torch.nn.Conv1d(in_channels=128, out_channels=256, kernel_size=7, padding=3),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(768, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, output_dim)
        )

    def forward(self, x):
        # Compute features using convolutional layers
        x = self.features(x)
        activation = [x]
        # Compute classification using fully-connected layers
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x, {"activation": activation}

    def reset_parameters(self):
        for component in [self.features, self.classifier]:
            for m in component:
                # Not all modules have parameters to reset
                try:
                    m.reset_parameters()
                except AttributeError:
                    pass
