import torch
import torch.nn as nn


class Upsamplelayer(nn.Module):
    def __init__(self, in_ch, out_ch, mode="bicubic"):
        super(Upsamplelayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2, mode=mode, align_corners=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        return self.layer(x)


class Decoder(nn.Module):
    """
    应力差解码器
    """
    def __init__(self):
        super(Decoder, self).__init__()
        self.layer1 = Upsamplelayer(512, 512)
        self.layer2 = Upsamplelayer(512, 256)
        self.layer3 = Upsamplelayer(256, 128)
        self.layer4 = Upsamplelayer(128, 64)
        self.layer5 = Upsamplelayer(64, 32)
        self.out = nn.Conv2d(32, 1, kernel_size=(1, 1))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        out = self.out(x)

        return torch.sigmoid(out)


if __name__ == "__main__":
    decoder_net = Decoder()
    ipt = torch.randn((1, 512, 7, 7))
    opt = decoder_net(ipt)
    print(opt.size())
