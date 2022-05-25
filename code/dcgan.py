import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torchvision
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from PIL import Image

imgs_path = glob.glob('./data/anime-faces/*.jpg')
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=0.5, std=0.5),
                                ])


# 读取图片
class Face_dataset(data.Dataset):
    def __init__(self, imgs_path):
        self.imgs_path = imgs_path

    def __getitem__(self, index):
        imgs_path = self.imgs_path[index]
        pil_img = Image.open(imgs_path)
        pil_img = transform(pil_img)
        return pil_img

    def __len__(self):
        return len(self.imgs_path)


dataset = Face_dataset(imgs_path)
BATCH_SIZE = 32
dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
imgs_batch = next(iter(dataloader))


# 定义生成器

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.linear1 = nn.Linear(100, 256 * 16 * 16)
        self.bn1 = nn.BatchNorm1d(256 * 16 * 16)
        self.deconv1 = nn.ConvTranspose2d(256, 128,
                                          kernel_size=(3, 3),
                                          stride=1,
                                          padding=1
                                          )  # 得到128*16*16的图像
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 64,
                                          kernel_size=(4, 4),
                                          stride=2,
                                          padding=1  # 64*32*32
                                          )
        self.bn3 = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(64, 3,
                                          kernel_size=(4, 4),
                                          stride=2,
                                          padding=1  # 3*64*64
                                          )

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.bn1(x)
        x = x.view(-1, 256, 16, 16)
        x = F.relu(self.deconv1(x))
        x = self.bn2(x)
        x = F.relu(self.deconv2(x))
        x = self.bn3(x)
        x = torch.tanh(self.deconv3(x))
        return x


# 定义判别器
# input:1*28*28
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.bn = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128 * 15 * 15, 1)

    def forward(self, x):
        x = F.dropout2d(F.leaky_relu(self.conv1(x)))  # dropout减轻判别器性能
        x = F.dropout2d(F.leaky_relu(self.conv2(x)))  # (batch, 128,15,15)
        x = self.bn(x)
        x = x.view(-1, 128 * 15 * 15)  # (batch, 128,15,15)--->  (batch, 128*15*15)
        x = torch.sigmoid(self.fc(x))
        return x


# 初始化模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
gen = Generator().to(device)
dis = Discriminator().to(device)

# 损失计算函数
loss_function = torch.nn.BCELoss()

# 定义优化器
d_optim = torch.optim.Adam(dis.parameters(), lr=0.00001)
g_optim = torch.optim.Adam(gen.parameters(), lr=0.0001)


def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input).permute(0, 2, 3, 1).cpu().numpy()
    fig = plt.figure(figsize=(12, 8))
    for i in range(predictions.shape[0]):
        plt.subplot(2, 4, i + 1)
        plt.imshow((predictions[i] + 1) / 2)
        plt.axis("off")

    plt.show()


test_input = torch.randn(8, 100, device=device)

# 开始训练
D_loss = []
G_loss = []
# 训练循环
for epoch in range(100):
    d_epoch_loss = 0
    g_epoch_loss = 0
    batch_count = len(dataloader)  # 返回的是批次数
    # 对全部的数据集做一次迭代
    for step, img in enumerate(dataloader):
        img = img.to(device)  # 上传到设备上
        size = img.shape[0]  # 返回img的第一维的大小
        random_noise = torch.randn(size, 100, device=device)

        d_optim.zero_grad()  # 将上述步骤的梯度归零
        real_output = dis(img)
        d_real_loss = loss_function(real_output,
                                    torch.ones_like(real_output, device=device)
                                    )

        # 得到判别器在生成图像上的损失
        gen_img = gen(random_noise)
        fake_output = dis(gen_img.detach())
        d_fake_loss = loss_function(fake_output,
                                    torch.zeros_like(fake_output, device=device))

        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        d_optim.step()  # 优化

        # 得到生成器的损失
        g_optim.zero_grad()
        fake_output = dis(gen_img)
        g_loss = loss_function(fake_output,
                               torch.ones_like(fake_output, device=device))
        g_loss.backward()
        g_optim.step()

        with torch.no_grad():
            d_epoch_loss += d_loss.item()
            g_epoch_loss += g_loss.item()
    with torch.no_grad():
        d_epoch_loss /= batch_count  # 平均loss
        g_epoch_loss /= batch_count
        D_loss.append(d_epoch_loss)
        G_loss.append(g_epoch_loss)
        generate_and_save_images(gen, epoch, test_input)
    print('Epoch:', epoch)
plt.plot(D_loss, label='D_loss')
plt.plot(G_loss, label='G_loss')
plt.legend()
