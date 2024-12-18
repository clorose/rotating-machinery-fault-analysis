# path: ~/Develop/rotating-machinery-fault-analysis/src/models/cnn.py
import torch
import torch.nn as nn

class KAMP_CNN(nn.Module):
    def __init__(self, input_channels=1, hidden_channels=100, num_classes=4):
        super(KAMP_CNN, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, 
                    out_channels=hidden_channels, 
                    kernel_size=2, 
                    stride=1, 
                    padding='same'),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1, stride=1),
            nn.Dropout(p=0.2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_channels, 
                    out_channels=num_classes, 
                    kernel_size=2, 
                    stride=1, 
                    padding='same'),
            nn.BatchNorm1d(num_classes),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1, stride=1)
        )
        
        self.final_pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(num_classes, num_classes)
        
    def forward(self, x):
        x = x.unsqueeze(1)  # 채널 차원 추가
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.final_pool(x)
        x = self.linear(x.squeeze(-1))
        
        return x