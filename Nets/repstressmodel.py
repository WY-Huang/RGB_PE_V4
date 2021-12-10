from Nets.encoder_repvgg import *
from Nets.decoder_stress import Decoder


class RepStressmodel(nn.Module):
    def __init__(self):
        super(RepStressmodel, self).__init__()
        self.encoder = create_RepVGG_A1()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)
