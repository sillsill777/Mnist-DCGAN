import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision import transforms
from tqdm import tqdm
import cv2
import os
from imutils import build_montages


def make_block(in_c, out_c, kernel, stride, padding):
    layer = []
    layer.append(nn.ConvTranspose2d(in_c, out_c, kernel, stride, padding, bias=False))
    layer.append(nn.ReLU())
    layer.append(nn.BatchNorm2d(out_c))
    return layer


class Generator(nn.Module):

    def __init__(self, noize_dim):
        super().__init__()
        self.noize_dim = noize_dim
        l1 = make_block(noize_dim, 128, 4, 1, 0)  # 4
        l2 = make_block(128, 64, 3, 2, 1)  # 7
        l3 = make_block(64, 32, 4, 2, 1)  # 14
        l4 = make_block(32, 1, 4, 2, 1)[:-2]  # 28
        self.layers = nn.Sequential(
            *l1, *l2, *l3, *l4, nn.Tanh())

    def forward(self, x):
        return self.layers(x)


class Discriminator(nn.Module):
    def __init__(self, alpha=0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.LeakyReLU(alpha),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.LeakyReLU(alpha),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.LeakyReLU(alpha),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

traindata = MNIST('./mnist', train=True, transform=transforms, download=True)
testdata = MNIST('./mnist', train=False, transform=transforms, download=True)
data = torch.utils.data.ConcatDataset((traindata, testdata))
dataLoader = torch.utils.data.DataLoader(data, batch_size=128, shuffle=True)
os.makedirs('img', exist_ok=True)


def weight_init(module):
    name = module.__class__.__name__
    if name.find("Conv") != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif name.find("Batch") != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)


trainstep = len(dataLoader.dataset) // 128
gen = Generator(100)
disc = Discriminator()
gen.apply(weight_init)
disc.apply(weight_init)
gen.to(DEVICE)
disc.to(DEVICE)

genOpt = torch.optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
discOpt = torch.optim.Adam(disc.parameters(), lr=0.0002, betas=(0.5, 0.999))

lossf = torch.nn.BCELoss()
benchmark = torch.randn((256, 100, 1, 1), device=DEVICE)

for epoch in tqdm(range(30), leave=False, desc='Epoch: '):
    totalLossG = 0
    totalLossD = 0
    data = iter(dataLoader)
    innerTqdm = tqdm(range(trainstep), leave=False, desc='Step: ')
    for steps in innerTqdm:
        (img, label) = next(data)
        img = img.to(DEVICE)

        disc.train()
        gen.train()

        disc.zero_grad()
        batch_size = img.shape[0]
        truelabel = torch.ones((batch_size), device=DEVICE)
        pred = disc(img).view(-1)
        trueloss = lossf(pred, truelabel)

        noize = torch.randn((batch_size, 100, 1, 1), device=DEVICE)

        fake = gen(noize)
        fakelabel = torch.zeros((batch_size), device=DEVICE)

        fakeloss = lossf(disc(fake.detach()).view(-1), fakelabel)
        loss = fakeloss + trueloss
        loss.backward()
        discOpt.step()

        gen.zero_grad()
        genloss = lossf(disc(fake).view(-1), truelabel)
        genloss.backward()
        genOpt.step()

        totalLossG += loss
        totalLossD += genloss
        innerTqdm.set_postfix({'Gen_loss': '{:.4f}'.format(genloss), 'Disc_loss': '{:.4f}'.format(loss)})

    with torch.no_grad():
        gen.eval()
        benchout = gen(benchmark)
        benchout = benchout.cpu().detach().permute((0, 2, 3, 1)).numpy()
        benchout = (benchout * 127.5 + 127.5).astype(np.uint8)
        img = np.repeat(benchout, 3, axis=-1)
        vis = build_montages(img, (28, 28), (16, 16))[0]
        path = "./img/{}.png".format(epoch + 1)
        cv2.imwrite(path, vis)
