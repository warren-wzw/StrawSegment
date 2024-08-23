from telnetlib import SE
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
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

class SegNet(nn.Module):
    def __init__(self, classes=4):
        super(SegNet, self).__init__()

        batchNorm_momentum = 0.1

        self.conv11 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(64, momentum=batchNorm_momentum)

        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(128, momentum=batchNorm_momentum)

        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)

        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)

        self.conv51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)

        self.conv53d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv52d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv51d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)

        self.conv43d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv42d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv41d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)

        self.conv33d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv32d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv31d = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(128, momentum=batchNorm_momentum)

        self.conv22d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        self.conv21d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(64, momentum=batchNorm_momentum)

        self.conv12d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.conv11d = nn.Conv2d(64, classes, kernel_size=3, padding=1)

    def forward(self, x):
        # Stage 1
        x11 = F.relu(self.bn11(self.conv11(x)))
        x12 = F.relu(self.bn12(self.conv12(x11)))
        x1_size = x12.size()
        x1p, id1 = F.max_pool2d(x12, kernel_size=2, stride=2, return_indices=True)

        # Stage 2
        x21 = F.relu(self.bn21(self.conv21(x1p)))
        x22 = F.relu(self.bn22(self.conv22(x21)))
        x2_size = x22.size()
        x2p, id2 = F.max_pool2d(x22, kernel_size=2, stride=2, return_indices=True)

        # Stage 3
        x31 = F.relu(self.bn31(self.conv31(x2p)))
        x32 = F.relu(self.bn32(self.conv32(x31)))
        x33 = F.relu(self.bn33(self.conv33(x32)))
        x3_size = x33.size()
        x3p, id3 = F.max_pool2d(x33, kernel_size=2, stride=2, return_indices=True)

        # Stage 4
        x41 = F.relu(self.bn41(self.conv41(x3p)))
        x42 = F.relu(self.bn42(self.conv42(x41)))
        x43 = F.relu(self.bn43(self.conv43(x42)))
        x4_size = x43.size()
        x4p, id4 = F.max_pool2d(x43, kernel_size=2, stride=2, return_indices=True)

        # Stage 5
        x51 = F.relu(self.bn51(self.conv51(x4p)))
        x52 = F.relu(self.bn52(self.conv52(x51)))
        x53 = F.relu(self.bn53(self.conv53(x52)))
        x5_size = x53.size()
        x5p, id5 = F.max_pool2d(x53, kernel_size=2, stride=2, return_indices=True)

        # Stage 5d
        x5d = F.max_unpool2d(x5p, id5, kernel_size=2, stride=2, output_size=x5_size)
        x53d = F.relu(self.bn53d(self.conv53d(x5d)))
        x52d = F.relu(self.bn52d(self.conv52d(x53d)))
        x51d = F.relu(self.bn51d(self.conv51d(x52d)))

        # Stage 4d
        x4d = F.max_unpool2d(x51d, id4, kernel_size=2, stride=2, output_size=x4_size)
        x43d = F.relu(self.bn43d(self.conv43d(x4d)))
        x42d = F.relu(self.bn42d(self.conv42d(x43d)))
        x41d = F.relu(self.bn41d(self.conv41d(x42d)))

        # Stage 3d
        x3d = F.max_unpool2d(x41d, id3, kernel_size=2, stride=2, output_size=x3_size)
        x33d = F.relu(self.bn33d(self.conv33d(x3d)))
        x32d = F.relu(self.bn32d(self.conv32d(x33d)))
        x31d = F.relu(self.bn31d(self.conv31d(x32d)))

        # Stage 2d
        x2d = F.max_unpool2d(x31d, id2, kernel_size=2, stride=2, output_size=x2_size)
        x22d = F.relu(self.bn22d(self.conv22d(x2d)))
        x21d = F.relu(self.bn21d(self.conv21d(x22d)))

        # Stage 1d
        x1d = F.max_unpool2d(x21d, id1, kernel_size=2, stride=2, output_size=x1_size)
        x12d = F.relu(self.bn12d(self.conv12d(x1d)))
        x11d = self.conv11d(x12d)

        return x11d