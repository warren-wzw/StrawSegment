from telnetlib import SE
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.transforms.functional as TF
from model.transformer import TransformerBottleneck

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
        
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        """Encoder"""
        e1 = self.encoder1(x) #1,32,256,256
        e2 = self.encoder2(F.max_pool2d(e1, 2))#1,64,128,128
        e3 = self.encoder3(F.max_pool2d(e2, 2))#1,128,64,64
        e4 = self.encoder4(F.max_pool2d(e3, 2))#1,256,32,32
        e5 = self.encoder5(F.max_pool2d(e4, 2))#1,512,16,16
        """Bottleneck"""
        b = self.bottleneck(F.max_pool2d(e5, 2))#1,1024.8,8

        """Decoder"""
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
    
class UNet_Se(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet_Se, self).__init__()
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
    
    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            SeparableConv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Softplus(),
            SeparableConv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Softplus()
        ) 
           
    def _upconv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        )
        
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        """Encoder"""
        e1 = self.encoder1(x) #1,32,256,256
        e2 = self.encoder2(F.max_pool2d(e1, 2))#1,64,128,128
        e3 = self.encoder3(F.max_pool2d(e2, 2))#1,128,64,64
        e4 = self.encoder4(F.max_pool2d(e3, 2))#1,256,32,32
        e5 = self.encoder5(F.max_pool2d(e4, 2))#1,512,16,16
        """Bottleneck"""
        b = self.bottleneck(F.max_pool2d(e5, 2))#1,1024.8,8

        """Decoder"""
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

class UNet_Atten(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet_Atten, self).__init__()
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
        self.attention5 = SelfAttention(512)
    
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
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        """Encoder"""
        e1 = self.encoder1(x) #1,32,256,256
        e2 = self.encoder2(F.max_pool2d(e1, 2))#1,64,128,128
        e3 = self.encoder3(F.max_pool2d(e2, 2))#1,128,64,64
        e4 = self.encoder4(F.max_pool2d(e3, 2))#1,256,32,32
        e5 = self.encoder5(F.max_pool2d(e4, 2))#1,512,16,16
        """Apply self-attention"""
        e5 = self.attention5(e5)
        """Bottleneck"""
        b = self.bottleneck(F.max_pool2d(e5, 2))#1,1024.8,8
         
        """Decoder"""
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
    
class UNet_Se_Atten(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet_Se_Atten, self).__init__()
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
        self.attention5 = SelfAttention(512)
    
    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            SeparableConv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Softplus(),
            SeparableConv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Softplus()
        ) 
    
    def _upconv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        )
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        """Encoder"""
        e1 = self.encoder1(x) #1,32,256,256
        e2 = self.encoder2(F.max_pool2d(e1, 2))#1,64,128,128
        e3 = self.encoder3(F.max_pool2d(e2, 2))#1,128,64,64
        e4 = self.encoder4(F.max_pool2d(e3, 2))#1,256,32,32
        e5 = self.encoder5(F.max_pool2d(e4, 2))#1,512,16,16
        """Apply self-attention"""
        e5 = self.attention5(e5)
        """Bottleneck"""
        b = self.bottleneck(F.max_pool2d(e5, 2))#1,1024.8,8
         
        """Decoder"""
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
                
class UNet_Se_Atten_Trans(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet_Se_Atten_Trans, self).__init__()
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
        self.attention5 = SelfAttention(512)
        self.transformer=TransformerBottleneck(512)
    
    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            SeparableConv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Softplus(),
            SeparableConv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Softplus()
        ) 
           
    def _block_s2(self, in_channels, out_channels):
        return nn.Sequential(
            SeparableConv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Softplus(),
            SeparableConv2d(out_channels, out_channels, kernel_size=3, stride=2,padding=1),
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
        """Encoder"""
        e1 = self.encoder1(x) #1,32,256,256
        e2 = self.encoder2(F.max_pool2d(e1, 2))#1,64,128,128
        e3 = self.encoder3(F.max_pool2d(e2, 2))#1,128,64,64
        e4 = self.encoder4(F.max_pool2d(e3, 2))#1,256,32,32
        e5 = self.encoder5(F.max_pool2d(e4, 2))#1,512,16,16
        """Apply self-attention"""
        e5 = self.attention5(e5)
        """Bottleneck"""
        #b = self.bottleneck(F.max_pool2d(e5, 2))#1,1024.8,8
        b1 = F.max_pool2d(e5, 2)
        b2 = self.transformer(b1)
        b  = torch.cat((b1,b2),dim=1)#1,1024.8,8
         
        """Decoder"""
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

class UNet_Atten_Trans(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet_Atten_Trans, self).__init__()
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
        self.attention5 = SelfAttention(512)
        self.transformer=TransformerBottleneck(512)
    
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
        """Encoder"""
        e1 = self.encoder1(x) #1,32,256,256
        e2 = self.encoder2(F.max_pool2d(e1, 2))#1,64,128,128
        e3 = self.encoder3(F.max_pool2d(e2, 2))#1,128,64,64
        e4 = self.encoder4(F.max_pool2d(e3, 2))#1,256,32,32
        e5 = self.encoder5(F.max_pool2d(e4, 2))#1,512,16,16
        """Apply self-attention"""
        e5 = self.attention5(e5)
        """Bottleneck"""
        #b = self.bottleneck(F.max_pool2d(e5, 2))#1,1024.8,8
        b1 = F.max_pool2d(e5, 2)
        b2 = self.transformer(b1)
        b  = torch.cat((b1,b2),dim=1)#1,1024.8,8
         
        """Decoder"""
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
    
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNet2(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=4, features=[64, 128, 256, 512],
    ):
        super(UNet2, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)  

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class DoubleSeConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleSeConv, self).__init__()
        self.conv = nn.Sequential(
            SeparableConv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SeparableConv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNet2(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=4, features=[64, 128, 256, 512],
    ):
        super(UNet2, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)  

class UNet2_se(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=4, features=[64, 128, 256, 512],
    ):
        super(UNet2_se, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleSeConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleSeConv(feature*2, feature))

        self.bottleneck = DoubleSeConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)  
    
class UNet2_Atten(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=4, features=[64, 128, 256, 512],
    ):
        super(UNet2_Atten, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.attention5 = SelfAttention(512)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x=self.attention5(x)
        
        x = self.bottleneck(x)
        
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)  
    
class UNet2_se_Atten(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=4, features=[64, 128, 256, 512],
    ):
        super(UNet2_se_Atten, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.attention5 = SelfAttention(512)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleSeConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleSeConv(feature*2, feature))

        self.bottleneck = DoubleSeConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x=self.attention5(x)
        
        x = self.bottleneck(x)
        
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)  

class UNet2_se_Atten_Trans(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=4, features=[64, 128, 256, 512],
    ):
        super(UNet2_se_Atten_Trans, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.attention5 = SelfAttention(512)
        self.transformer=TransformerBottleneck(512)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleSeConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleSeConv(feature*2, feature))

        self.bottleneck = DoubleSeConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x=self.attention5(x)
        
        #x = self.bottleneck(x)
        b1 = x
        b2 = self.transformer(x)
        x  = torch.cat((b1,b2),dim=1)#1,1024.8,8
        
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)  

class UNet2_Atten_Trans(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=4, features=[64, 128, 256, 512],
    ):
        super(UNet2_Atten_Trans, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.attention5 = SelfAttention(512)
        self.transformer=TransformerBottleneck(512)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x=self.attention5(x)
        
        #x = self.bottleneck(x)
        b1 = x
        b2 = self.transformer(x)
        x  = torch.cat((b1,b2),dim=1)#1,1024.8,8
        
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)  