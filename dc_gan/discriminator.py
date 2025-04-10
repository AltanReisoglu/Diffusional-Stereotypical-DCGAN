import torch 
import numpy as np
from torch import nn
from torch.nn import functional as F



class Discriminator(nn.Module):


    def __init__(self, num_ch, num_disc_filter):
        super(Discriminator, self).__init__()
        self.network = nn.Sequential(
             nn.Conv2d(
                in_channels=num_ch,
                out_channels=num_disc_filter/2,#64
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.Conv2d(
                in_channels=num_disc_filter/2,
                out_channels=num_disc_filter,#64
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(#128
                in_channels=num_disc_filter,
                out_channels=num_disc_filter * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_disc_filter * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(#256
                in_channels=num_disc_filter * 2,
                out_channels=num_disc_filter * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_disc_filter * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(#512
                in_channels=num_disc_filter * 4,
                out_channels=num_disc_filter * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_disc_filter * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=num_disc_filter * 8,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.Sigmoid()
        )

    # The Discriminator outputs a scalar probability to classify the input image as real or fake.
    def forward(self, input):
        output = self.network(input)
        return output
nc=3
ndf=64
class Discriminator2(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super(Discriminator2, self).__init__()
        
        self.main = nn.Sequential(
            # Input: (nc) x 128 x 128
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),       # → (ndf) x 64 x 64
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),  # → (ndf*2) x 32 x 32
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),  # → (ndf*4) x 16 x 16
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),  # → (ndf*8) x 8 x 8
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False),  # → (ndf*8) x 4 x 4
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),        # → 1 x 1 x 1
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(-1, 1)

    
