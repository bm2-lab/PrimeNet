import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        reduced_channels = max(1, in_channels // reduction_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        return x * self.sigmoid(avg_out)

class Conv_Attention(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super(Conv_Attention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        # self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False)
    def forward(self, x):
        # return self.conv2(x) * self.sigmoid(self.conv1(x))
        return x * self.sigmoid(self.conv1(x))

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 1), stride=(1, 1)):
        super(ResidualConvBlock, self).__init__()
        # 第一层卷积
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=(1, 0))
        self.relu = nn.ReLU(inplace=True)
        # 第二层卷积
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), stride=(1, 1))       
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        # 计算残差
        residual = self.residual(x)
        # 卷积过程
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        # 返回卷积后的输出与残差相加
        return out + residual

# 多尺度卷积层
class MultiScaleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConv2d, self).__init__()
        self.conv1x2 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 2), padding=(0, 0))
        self.attn1x2 = Conv_Attention(out_channels) # 添加注意力机制
        self.conv3x2 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 2), padding=(1, 0))
        self.attn3x2 = Conv_Attention(out_channels) # 添加注意力机制
        self.conv9x2 = nn.Conv2d(in_channels, out_channels, kernel_size=(9, 2), padding=(4, 0))
        self.attn9x2 = Conv_Attention(out_channels) # 添加注意力机制
        conv_num = 3    
        self.attn_cat = ChannelAttention(conv_num * out_channels)
        
        # 一个额外的卷积层来将输出通道数调整为你需要的通道数
        self.adjust_channels = nn.Conv2d(conv_num * out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # 分别通过不同尺度的卷积层
        out1x2 = self.conv1x2(x)
        out1x2 = self.attn1x2(out1x2)
        out3x2 = self.conv3x2(x)
        out3x2 = self.attn3x2(out3x2)
        out9x2 = self.conv9x2(x)
        out9x2 = self.attn9x2(out9x2)
        
        # 拼接后调整通道数
        concat_out = torch.cat([out1x2, out3x2, out9x2], dim=1)
        concat_out = self.attn_cat(concat_out)
        return self.adjust_channels(concat_out)

class PrimeNet(nn.Module):
    def __init__(self):
        super(PrimeNet, self).__init__()
        
        # 修改第一个卷积层为多尺度卷积层
        self.conv1 = MultiScaleConv2d(8, 14*8)
        self.conv2 = ResidualConvBlock(14*8, 18*8)
        self.conv3 = ResidualConvBlock(18*8, 18*8)

        # 调整归一化层的维度
        self.norm1 = nn.LayerNorm([14*8, 128, 1])
        self.norm2 = nn.LayerNorm([18*8, 64, 1])
        self.norm3 = nn.LayerNorm([18*8, 32, 1])

        # 注意力模块
        self.attn0 = Conv_Attention(8)
        self.attn1 = Conv_Attention(14*8)
        self.attn2 = Conv_Attention(18*8)
        self.attn3 = Conv_Attention(18*8)
        self.attn4 = ChannelAttention(18*8)

        # Dropout层
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.002)
        self.dropout3 = nn.Dropout(0.2)

        self.pool = nn.MaxPool2d(kernel_size=(2, 1))

        # 第一层全连接层，共享
        self.fc1_shared = nn.Sequential(
            nn.Linear(18*8*32, 800),
            nn.ReLU(),
            nn.Dropout(0.15)
        )

        # 独立的后续全连接层
        self.fc1_branch1 = nn.Sequential(
            nn.Linear(800, 192),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(192, 1)
        )
        self.fc1_branch2 = nn.Sequential(
            nn.Linear(800, 192),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(192, 1)
        )
        self.fc1_branch3 = nn.Sequential(
            nn.Linear(800, 192),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(192, 1)
        )

    def forward(self, x):
        x = self.attn0(x)
        x = self.conv1(x)
        x = nn.ReLU(inplace=False)(x)
        x = self.norm1(x)
        x = self.dropout1(x)
        x = self.attn1(x)

        x = self.conv2(x)
        x = nn.ReLU(inplace=False)(x)
        x = self.pool(x)
        x = self.norm2(x)
        x = self.dropout2(x)
        x = self.attn2(x)

        x = self.conv3(x)
        x = nn.ReLU(inplace=False)(x)
        x = self.pool(x)
        x = self.norm3(x)
        x = self.dropout3(x)
        x = self.attn3(x)
        x = self.attn4(x)

        x = x.view(x.size(0), -1)  # Flatten
        
        # Shared fully connected layer
        x = self.fc1_shared(x)
        
        # Independent branches
        out1 = self.fc1_branch1(x)
        out2 = self.fc1_branch2(x)
        out3 = self.fc1_branch3(x)

        return torch.cat([out1, out2, out3], dim=1)