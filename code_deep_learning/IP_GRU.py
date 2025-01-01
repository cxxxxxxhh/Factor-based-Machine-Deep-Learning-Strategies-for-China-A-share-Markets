import torch
import torch.nn as nn
import torch.nn.init as init
import math

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)
        self._initialize_weights()

    def _initialize_weights(self):
        # 遍历 GRU 层中的所有参数并进行初始化
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:  # 输入到隐藏的权重 (input-to-hidden weights)
                init.xavier_uniform_(param, gain=1.0)  # 使用 Xavier 均匀分布初始化，gain=1.0保证均匀性
            elif 'weight_hh' in name:  # 隐藏到隐藏的权重 (hidden-to-hidden weights)
                init.kaiming_uniform_(param, a=math.sqrt(5), nonlinearity='relu')  # 使用 He 初始化，nonlinearity 为 ReLU
            elif 'bias' in name:  # 偏置
                init.constant_(param, 0)  # 偏置初始化为0
        
    def forward(self, x):
        # Initialize hidden state for GRU
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # Forward propagate through GRU
        out, _ = self.gru(x, h0)
        # Apply dropout and fully connected layer on the last time step's output
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out