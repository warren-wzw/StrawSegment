import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = SeparableConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = SeparableConv2d(out_channels, out_channels, kernel_size, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)  # 残差连接
        out = self.relu(out)
        return out

class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h).view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
        k = self.proj_k(h).view(B, C, -1)  # (B, C, H*W)
        v = self.proj_v(h).view(B, C, -1)  # (B, C, H*W)
        
        w = torch.bmm(q, k) * (C ** -0.5)  # (B, H*W, H*W)
        w = F.softmax(w, dim=-1)
        
        h = torch.bmm(v, w)  # (B, C, H*W)
        h = h.view(B, C, H, W)  # (B, C, H, W)
        h = self.proj(h)
        
        return x + h

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        batch_size, C, height, width = x.size()
        
        # Query, Key, Value
        Q = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # (B, N, C//8)
        K = self.key_conv(x).view(batch_size, -1, height * width)  # (B, C//8, N)
        V = self.value_conv(x).view(batch_size, -1, height * width)  # (B, C, N)
        
        # Attention
        attention = F.softmax(torch.bmm(Q, K), dim=1)  # (B, N, N)
        out = torch.bmm(V, attention.permute(0, 2, 1))  # (B, C, N)
        
        out = out.view(batch_size, C, height, width)
        out = self.gamma * out + x
        
        return out
       
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self._initialize_weights()
        # Define the blocks
        self.encoder1 = self._block(in_channels, 32)
        self.encoder2 = self._block(32, 64)
        self.encoder3 = self._block(64, 128)
        self.encoder4 = self._block(128, 256)
        self.encoder5 = self._block(256, 512)
        
        self.bottleneck = self._block(512, 1024)
        
        self.upconv5 = self._upconv(1024, 512)
        self.decoder5 = self._block(1024, 512)
        self.upconv4 = self._upconv(512, 256)
        self.decoder4 = self._block(512, 256) 
        self.upconv3 = self._upconv(256, 128)
        self.decoder3 = self._block(256, 128)
        self.upconv2 = self._upconv(128, 64)
        self.decoder2 = self._block(128, 64)
        self.upconv1 = self._upconv(64, 32)
        self.decoder1 = self._block(64, 32)
        
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)
        
        # Add Self-Attention modules
        self.attention5 = SelfAttention(512)
        self.attention4 = SelfAttention(256)
        self.attention3 = SelfAttention(128)
        self.attention2 = SelfAttention(64)
        
    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Softplus(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Softplus()
        )
    
    def _upconv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        )
        
    def _upconv_pix(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2)
        )
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(F.max_pool2d(e1, 2))
        e3 = self.encoder3(F.max_pool2d(e2, 2))
        e4 = self.encoder4(F.max_pool2d(e3, 2))
        e5 = self.encoder5(F.max_pool2d(e4, 2))
        
        # Apply self-attention
        e5 = self.attention5(e5)
        e4 = self.attention4(e4)
        e3 = self.attention3(e3)
        e2 = self.attention2(e2)
        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e5, 2))
        
        # Decoder
        d5 = torch.cat((self.upconv5(b), e5), dim=1)
        d5 = self.decoder5(d5) 
        d4 = torch.cat((self.upconv4(d5), e4), dim=1)
        d4 = self.decoder4(d4)
        d3 = torch.cat((self.upconv3(d4), e3), dim=1)
        d3 = self.decoder3(d3)
        d2 = torch.cat((self.upconv2(d3), e2), dim=1)
        d2 = self.decoder2(d2)
        d1 = torch.cat((self.upconv1(d2), e1), dim=1)
        d1 = self.decoder1(d1)

        out = self.final_conv(d1)
        return out  # Directly return the output

