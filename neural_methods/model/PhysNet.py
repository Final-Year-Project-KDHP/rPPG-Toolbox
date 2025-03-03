import torch
import torch.nn as nn

class PhysNet(nn.Module):
    def __init__(self, frames=60):
        super(PhysNet, self).__init__()
        
        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 16, (1, 5, 5), stride=1, padding=(0, 2, 2)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, (3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        
        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, (3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        
        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, (3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        
        self.ConvBlock5 = nn.Sequential(
            nn.Conv3d(64, 64, (3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        
        self.ConvBlock6 = nn.Sequential(
            nn.Conv3d(64, 64, (3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(64, 64, (4, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )
        
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, (4, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )
        
        self.ConvBlock10 = nn.Conv3d(64, 1, (1, 1, 1), stride=1, padding=0)

        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)
        self.poolspa = nn.AdaptiveAvgPool3d((frames, 1, 1))
        
        # LSTM layers
        self.lstm_1 = nn.LSTM(60, 32, batch_first=True, bidirectional=True)
        self.lstm_2 = nn.LSTM(64, 24, batch_first=True, bidirectional=True)
        self.lstm_3 = nn.LSTM(48, 8, batch_first=True, bidirectional=True)
        self.lstm_4 = nn.LSTM(16, 1, batch_first=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(60, 32)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(32, 1)
        
    def forward(self, x):
        batch_size, channel, length, width, height = x.shape

        x = self.ConvBlock1(x)
        x = self.MaxpoolSpa(x)
        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)
        x = self.MaxpoolSpaTem(x)
        x = self.ConvBlock4(x)
        x = self.ConvBlock5(x)
        x = self.MaxpoolSpaTem(x)
        x = self.ConvBlock6(x)
        x = self.upsample(x)
        x = self.upsample2(x)
        x = self.poolspa(x)
        x = self.ConvBlock10(x)

        rPPG = x.view(batch_size, length)
        
        # LSTM layers
        x, _ = self.lstm_1(rPPG.unsqueeze(-1))
        x, _ = self.lstm_2(x)
        x, _ = self.lstm_3(x)
        x, _ = self.lstm_4(x)
        
        x = x.squeeze(-1)
        x = torch.tanh(self.fc1(x))
        x = self.dropout(x)
        HR_out = torch.relu(self.fc2(x))
        
        return rPPG, HR_out