import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images", exist_ok=True)

img_shape = (1, 28, 28)
smooth = 0.1  # 标签平滑

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(128, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

# Loss function
#
# function: - [y log p + (1 - y) log (1 - p)]
# label = 1: - log p
# label = 0: - log(1 - p)
# 对于 discriminater : label 为 1 时 bce loss 越小越好，label 为 0 时，p 越小越好，所以 loss 越小越好
# 对于 generater: 希望 label 为 1， 则 loss = - log p
adversarial_loss = torch.nn.BCELoss()

# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(28), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=2,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

Tensor = torch.FloatTensor

# ----------
#  Training
# ----------

for epoch in range(10):
    for i, (imgs, _) in enumerate(dataloader):

        valid = torch.full(size=(imgs.shape[0], 1), fill_value=1.0).float()
        fake = torch.full(size=(imgs.shape[0], 1), fill_value=0.0).float()

        # Configure input
        real_imgs = imgs.type(Tensor)

        # Sample noise as generator input
        z = Tensor(np.random.normal(0, 1, (imgs.shape[0], 128)))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid * (1 - smooth))
        fake_loss = adversarial_loss(discriminator(generator(z)), fake)
        d_loss = real_loss + fake_loss

        d_loss.backward()
        optimizer_D.step()

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = - adversarial_loss(1 - discriminator(gen_imgs), valid * (1 - smooth))
        # g_loss_ = torch.mean(torch.log(1 - discriminator(gen_imgs)))

        g_loss.backward()
        optimizer_G.step()

        batches_done = epoch * len(dataloader) + i
        if batches_done % 400 == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)


        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, 10, i, len(dataloader), d_loss.item(), g_loss.item())
        )