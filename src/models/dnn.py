# path: ~/Develop/rotating-machinery-fault-analysis/src/models/dnn.py
import torch
import torch.nn as nn

class KAMP_DNN(nn.Module):
    def __init__(self, input_size=4, hidden_size=100, num_classes=4, dropout_rate=0.2):
        super(KAMP_DNN, self).__init__()
        
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.layer4 = nn.Linear(hidden_size, num_classes)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer4(x)
        return x
