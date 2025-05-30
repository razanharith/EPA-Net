import torch
import torch.nn as nn
import torchvision.models as models


class UNetVGG16(nn.Module):
    def __init__(self, num_classes):
        super(UNetVGG16, self).__init__()

        # Load pre-trained VGG16 model
        vgg16 = models.vgg16(pretrained=True)

        # Extract the encoder part of VGG16
        self.encoder = nn.Sequential(*list(vgg16.features.children())[:-1])

        # Define the decoder part of U-Net
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # Encoder
        x1 = self.encoder[0:4](x)
        x2 = self.encoder[4:9](x1)
        x3 = self.encoder[9:16](x2)
        x4 = self.encoder[16:23](x3)
        x5 = self.encoder[23:](x4)

        # Decoder
        x = self.decoder[0](x5)
        x = torch.cat([x, x4], dim=1)
        x = self.decoder[1:3](x)
        x = self.decoder[3](x)
        x = torch.cat([x, x3], dim=1)
        x = self.decoder[4:6](x)
        x = self.decoder[6](x)
        x = torch.cat([x, x2], dim=1)
        x = self.decoder[7:9](x)
        x = self.decoder[9](x)
        x = torch.cat([x, x1], dim=1)
        x = self.decoder[10:](x)

        return x
