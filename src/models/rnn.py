# path: ~/Develop/rotating-machinery-fault-analysis/src/models/rnn.py
import torch
import torch.nn as nn

class KAMP_RNN(nn.Module):
    def __init__(self, input_size=4, hidden_size=100, num_layers=2, 
                num_classes=4, dropout_rate=0.2):
        super(KAMP_RNN, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate
        )
        
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = x.unsqueeze(1)  # 배치 우선 차원 추가
        
        # LSTM 출력: (output, (h_n, c_n))
        output, _ = self.lstm(x)
        
        # 마지막 타임스텝의 출력을 사용
        x = output.view(-1, output.size(-1))
        x = self.fc(x)
        
        return x
