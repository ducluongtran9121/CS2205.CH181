import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, name, in_channels, hidden_channels, num_hiddens, num_classes):
        super(CNN, self).__init__()
        self.name = name
        # 1st convolutional layer
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, padding='same')    # 116*1 --> 116*64
        self.relu1 = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm1d(num_features=hidden_channels)

        # # 2nd convolutional layer 
        self.conv2 = nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels*2, kernel_size=3)                  # 116*64 --> 112*128
        self.relu2 = nn.ReLU()
        self.batchnorm2 = nn.BatchNorm1d(num_features=hidden_channels*2)

        # Fully Connected layer
        self.fc1 = nn.Linear(in_features=68*hidden_channels*2, out_features=num_hiddens) #CIC
        # self.fc1 = nn.Linear(in_features=113*hidden_channels*2, out_features=num_hiddens) 
        # self.fc1 = nn.Linear(in_features=int(57.5*hidden_channels*2), out_features=num_hiddens) #NB
        # self.fc1 = nn.Linear(in_features=35*hidden_channels*2, out_features=num_hiddens) #CIC
        # self.fc1 = nn.Linear(in_features=21*hidden_channels*2, out_features=num_hiddens) #Edge
        self.fc2 = nn.Linear(in_features=num_hiddens, out_features=num_classes)
    
    def forward(self,x):
        
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.batchnorm1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.batchnorm2(x)

        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

class LeNet(nn.Module):
    def __init__(self, name, in_channels, hidden_channels, num_hiddens, num_classes):
        super(LeNet, self).__init__()
        self.name = name

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=2, stride=1, padding='same'),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 1))
        self.layer2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels*2, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(hidden_channels*2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 1))
        self.fc = nn.Linear(10*hidden_channels*2, num_hiddens)
        # self.fc = nn.Linear(22*hidden_channels*2, num_hiddens)
        # self.fc = nn.Linear(40*hidden_channels*2, num_hiddens)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(num_hiddens, num_hiddens//2)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(num_hiddens//2, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = torch.flatten(out,1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out
    
class LSTM(nn.Module):
    def __init__(self, name, in_channels, hidden_channels, num_hiddens, num_classes):
        super(LSTM, self).__init__()
        self.name = name
        self.input_size = 115
        self.hidden_size = num_hiddens
        self.num_layers = 2
        self.lstm = nn.LSTM(input_size=115, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x.reshape(x.shape[0], 1, x.shape[1]), (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class AE(nn.Module):
    def __init__(self) -> None:
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(64, 512),
            # nn.Linear(8704, 512),
            # nn.Linear(7360, 512),
            # nn.Linear(2560, 512),
            # nn.Linear(4480, 512),
            # nn.Linear(14464, 512),
            nn.ReLU(True),
            nn.Linear(512, 128),
            nn.ReLU(True), 
            nn.Linear(128, 64), 
            nn.ReLU(True), 
            nn.Linear(64, 16)
            )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 512),
            nn.ReLU(True), 
            nn.Linear(512, 64), 
            # nn.Linear(512, 7360), 
            # nn.Linear(512, 2560), 
            # nn.Linear(512, 4480), 
            # nn.Linear(512, 8704), 
            # nn.Linear(512, 14464), 
            nn.Tanh()
            )
    
    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x
    
