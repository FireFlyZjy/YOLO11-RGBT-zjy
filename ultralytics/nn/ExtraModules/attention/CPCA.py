import torch
import torch.nn as nn
import torch.nn.functional as F

class CPCA_ChannelAttention(nn.Module):
    def __init__(self, input_channels, internal_neurons):
        super().__init__()
        self.fc1 = nn.Conv2d(input_channels, internal_neurons, 1, 1, 0, bias=True)
        self.fc2 = nn.Conv2d(internal_neurons, input_channels, 1, 1, 0, bias=True)

    def forward(self, inputs):
        x1 = F.adaptive_avg_pool2d(inputs, (1, 1))
        x1 = self.fc1(x1); x1 = F.relu(x1); x1 = self.fc2(x1)
        x2 = F.adaptive_max_pool2d(inputs, (1, 1))
        x2 = self.fc1(x2); x2 = F.relu(x2); x2 = self.fc2(x2)
        return inputs * torch.sigmoid(x1 + x2)


class CPCA(nn.Module):
    """CPCA: Channel Prior Convolutional Attention, 通道先验+多尺度条带深度卷积"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        internal_neurons = max(1, channels // reduction)
        self.ca = CPCA_ChannelAttention(channels, internal_neurons)
        self.dconv5_5 = nn.Conv2d(channels, channels, 5, 1, 2, groups=channels)
        self.dconv1_7 = nn.Conv2d(channels, channels, (1, 7), 1, (0, 3), groups=channels)
        self.dconv7_1 = nn.Conv2d(channels, channels, (7, 1), 1, (3, 0), groups=channels)
        self.dconv1_11 = nn.Conv2d(channels, channels, (1, 11), 1, (0, 5), groups=channels)
        self.dconv11_1 = nn.Conv2d(channels, channels, (11, 1), 1, (5, 0), groups=channels)
        self.dconv1_21 = nn.Conv2d(channels, channels, (1, 21), 1, (0, 10), groups=channels)
        self.dconv21_1 = nn.Conv2d(channels, channels, (21, 1), 1, (10, 0), groups=channels)
        self.conv_out = nn.Conv2d(channels, channels, 1)
        self.act = nn.GELU()

    def forward(self, inputs):
        inputs = self.act(self.conv_out(inputs))
        inputs = self.ca(inputs)
        x_init = self.dconv5_5(inputs)
        x_1 = self.dconv7_1(self.dconv1_7(x_init))
        x_2 = self.dconv11_1(self.dconv1_11(x_init))
        x_3 = self.dconv21_1(self.dconv1_21(x_init))
        x = x_1 + x_2 + x_3 + x_init
        out = self.conv_out(x) * inputs
        return self.conv_out(out)
