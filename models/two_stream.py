import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoStream(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super(TwoStream, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels if inter_channels else in_channels // 8

        self.query_conv = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.spatial_out_conv = nn.Conv2d(self.inter_channels, in_channels, kernel_size=1)

        self.channel_query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.channel_key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.channel_value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.beta = nn.Parameter(torch.zeros(1))

        self.fc = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        batch_size, C, H, W = x.size()

        # ---- Spatial Attention ----
        proj_query = self.query_conv(x).view(batch_size, self.inter_channels, -1)
        proj_key = self.key_conv(x).view(batch_size, self.inter_channels, -1)
        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)
        attention = F.softmax(energy, dim=-1)                                      # Spatial attention map
        proj_value = self.value_conv(x).view(batch_size, self.inter_channels, -1)
        out_s = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out_s = out_s.view(batch_size, self.inter_channels, H, W)
        out_s = self.spatial_out_conv(out_s)
        F_s = self.gamma * out_s + x

        # ---- Channel Attention ----
        proj_query_c = self.channel_query_conv(x).view(batch_size, C, -1)
        proj_key_c = self.channel_key_conv(x).view(batch_size, C, -1)
        energy_c = torch.bmm(proj_query_c, proj_key_c.permute(0, 2, 1))
        attention_c = F.softmax(energy_c, dim=-1)                                  # Channel attention map
        proj_value_c = self.channel_value_conv(x).view(batch_size, C, -1)
        out_c = torch.bmm(attention_c, proj_value_c)
        out_c = out_c.view(batch_size, C, H, W)
        F_c = self.beta * out_c + x

        F_sc = F_s + F_c

        return F_sc
