import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvBlock(nn.Module):
    """
    A convolutional block with two convolutions and a residual connection.
    This "improves" the standard U-Net block.
    """

    def __init__(self, in_channels, out_channels):
        super(ResidualConvBlock, self).__init__()

        # First convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Residual connection
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = self.residual(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Add the residual connection
        out += identity
        return F.relu(out)


class EncoderBlock(nn.Module):
    """
    Encoder block (Down-sampling path)
    """

    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = ResidualConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)
        down = self.pool(skip)
        return down, skip


class DecoderBlock(nn.Module):
    """
    Decoder block (Up-sampling path)
    """

    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ResidualConvBlock(
            in_channels, out_channels
        )  # Note: in_channels = out_channels + skip_channels

    def forward(self, x, skip):
        up = self.up(x)

        # Concatenate skip connection
        combined = torch.cat([up, skip], dim=1)
        out = self.conv(combined)
        return out


class ImprovedUNet(nn.Module):
    """
    The main Improved U-Net model.
    """

    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(ImprovedUNet, self).__init__()

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        # Encoder path
        for feature in features:
            self.encoders.append(EncoderBlock(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = ResidualConvBlock(features[-1], features[-1] * 2)

        # Decoder path
        for feature in reversed(features):
            self.decoders.append(DecoderBlock(feature * 2, feature))

        # Final output layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encode
        for encoder in self.encoders:
            x, skip = encoder(x)
            skip_connections.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decode
        skip_connections = skip_connections[::-1]  # Reverse for decoding
        for i in range(len(self.decoders)):
            x = self.decoders[i](x, skip_connections[i])

        # Final output
        return self.final_conv(x)


if __name__ == "__main__":
    # Test if the model builds
    x = torch.randn((1, 1, 256, 256))  # Batch_size, channels, H, W
    model = ImprovedUNet(in_channels=1, out_channels=1)
    pred = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {pred.shape}")
