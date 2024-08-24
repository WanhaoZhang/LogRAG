import torch
import torch.nn as nn
import torch.nn.functional as F

from prelogad.DeepSVDD.src.base.base_net import BaseNet


class MLP(BaseNet):

    def __init__(self, num_i, num_h, num_o):
        super().__init__()
        self.rep_dim = num_o
        
        self.linear1 = torch.nn.Linear(num_i, num_h)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(num_h, num_h)  # 2个隐层
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(num_h, num_o)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x

class MLP_Autoencoder(BaseNet):

    def __init__(self, num_i, num_h, num_o):
        super().__init__()

        self.rep_dim = num_o

        # Encoder (must match the MLP network above)
        self.linear1 = torch.nn.Linear(num_i, num_h)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(num_h, num_h)  # 2个隐层
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(num_h, num_o)

        # Decoder
        self.deRelu = torch.nn.ReLU()
        self.deLinear1 = torch.nn.Linear(num_o, num_h)
        self.deRelu2 = torch.nn.ReLU()
        self.deLinear2 = torch.nn.Linear(num_h, num_h)  # 2个隐层
        self.deRelu3 = torch.nn.ReLU()
        self.deLinear3 = torch.nn.Linear(num_h, num_i)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.deRelu(x)
        x = self.deLinear1(x)
        x = self.deRelu2(x)
        x = self.deLinear2(x)
        x = self.deRelu3(x)
        x = self.deLinear3(x)
        return x