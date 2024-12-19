import torch
import torch.nn as nn

class KAMP_DNN(nn.Module):
    def __init__(self, input_size=4, hidden_size=32, num_classes=4, dropout_rate=0.3):
        super(KAMP_DNN, self).__init__()
        
        # Layer 1
        self.layer1 = nn.Linear(input_size, hidden_size)
        
        # Layer 2
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        
        # Layer 3
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        
        # Output layer
        self.layer4 = nn.Linear(hidden_size, num_classes)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Layer 1
        x = self.layer1(x)
        x = self.relu(x)
        
        # Layer 2
        x = self.layer2(x)
        x = self.relu(x)
        
        # Layer 3
        x = self.layer3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Output layer
        x = self.layer4(x)
        return x