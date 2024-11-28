import torch
import torch.nn as nn
import random
import numpy as np


# class for the float32 and int8 model
class QuantStubModel(nn.Module):
    def __init__(self, q=False):
        super(QuantStubModel, self).__init__()
        self.q = q
        if self.q:
            self.quant = torch.quantization.QuantStub()
            self.dequant = torch.quantization.DeQuantStub()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        if self.q:
            x = self.quant(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.q:
            x = self.dequant(x)
        return x


def calculate_model_size(model):
    param_size = sum(param.nelement() * param.element_size() for param in model.parameters())
    buffer_size = sum(buffer.nelement() * buffer.element_size() for buffer in model.buffers())
    total_size = param_size + buffer_size
    size_in_mb = total_size / (1024 ** 2)
    return total_size, size_in_mb