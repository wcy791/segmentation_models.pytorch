
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class ConvBlock(nn.Module):
    """A simple convolutional block."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class WaveletUnet(nn.Module):
    def __init__(self, encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=5):
        super().__init__()
        
        # --- Image Branch Encoder ---
        self.img_encoder = smp.encoders.get_encoder(
            encoder_name, 
            in_channels=in_channels, 
            depth=5, 
            weights=encoder_weights
        )
        # Channel sizes for resnet34: [3, 64, 64, 128, 256, 512]

        # --- Wavelet Branch Encoder (Aligned with Image Branch) ---
        # Input wavelet features are 256x256
        self.wavelet_conv1 = ConvBlock(3, 64)       # Out: 256x256
        self.wavelet_pool1 = nn.MaxPool2d(2)        # Out: 128x128
        self.wavelet_conv2 = ConvBlock(64, 128)     # Out: 128x128
        self.wavelet_pool2 = nn.MaxPool2d(2)        # Out: 64x64
        self.wavelet_conv3 = ConvBlock(128, 256)    # Out: 64x64
        self.wavelet_pool3 = nn.MaxPool2d(2)        # Out: 32x32
        self.wavelet_conv4 = ConvBlock(256, 256)    # Out: 32x32 (Changed from 512 to 256 for lighter branch)

        # --- Decoder (Built from scratch) ---
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.center = ConvBlock(in_channels=512, out_channels=512)

        # Decoder stages with corrected fusion channels
        self.fusion_conv1 = ConvBlock(512 + 256 + 256, 256) # Up(512) + Img(256) + Wavelet(256)
        self.fusion_conv2 = ConvBlock(256 + 128 + 256, 128) # Up(256) + Img(128) + Wavelet(256)
        self.fusion_conv3 = ConvBlock(128 + 64 + 128, 64)   # Up(128) + Img(64) + Wavelet(128)
        self.fusion_conv4 = ConvBlock(64 + 64 + 64, 32)     # Up(64) + Img(64) + Wavelet(64)
        self.final_conv = ConvBlock(32, 16)

        self.segmentation_head = nn.Conv2d(16, classes, kernel_size=1)

    def forward(self, image, wavelet_features):
        # --- Encode ---
        img_features = self.img_encoder(image)
        
        # Wavelet branch skips
        w1 = self.wavelet_conv1(wavelet_features) # 256x256
        w1_p = self.wavelet_pool1(w1)            # 128x128
        w2 = self.wavelet_conv2(w1_p)            # 128x128
        w2_p = self.wavelet_pool2(w2)            # 64x64
        w3 = self.wavelet_conv3(w2_p)            # 64x64
        w3_p = self.wavelet_pool3(w3)            # 32x32
        w4 = self.wavelet_conv4(w3_p)            # 32x32

        # --- Decode with Fusion ---
        x = self.center(img_features[-1]) # Deepest feature: 16x16

        # Upsample to 32x32
        x = self.upsample(x)
        x = torch.cat([x, img_features[-2], w4], dim=1)
        x = self.fusion_conv1(x)

        # Upsample to 64x64
        x = self.upsample(x)
        x = torch.cat([x, img_features[-3], w3], dim=1)
        x = self.fusion_conv2(x)

        # Upsample to 128x128
        x = self.upsample(x)
        x = torch.cat([x, img_features[-4], w2], dim=1)
        x = self.fusion_conv3(x)

        # Upsample to 256x256
        x = self.upsample(x)
        x = torch.cat([x, img_features[-5], w1], dim=1)
        x = self.fusion_conv4(x)
        
        # Final upsample to 512x512
        x = self.upsample(x)
        x = self.final_conv(x)
        
        logits = self.segmentation_head(x)
        
        return logits
