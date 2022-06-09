from random import randint
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
import shutil
import cv2
import random
from PIL import Image

image_size = 64
batch_size = 1

"""
生成器模型 
e1: 第一次卷积输出，输入为 边缘图 + 噪声图 
e2: 第二次卷积输出，输入为 e1
e3: 第三次卷积输出，输入为 e2
e4: 第四次卷积输出，输入为 e3
e5: 第五次卷积输出，输入为 e4 

d4: 第一次反卷积，输入为 e5
d4 = d4 + e4
d5: 第二次反卷积，输入为 d4
d5 = d5 + e3 即考虑e4的特征
d6: 第三次反卷积，输入为 d5
d6 = d6 + e2
d7: 第四次反卷积，输入为 d6
d7 = d7 + e1
d8: 最后一次反卷积
tanh(t8)
"""

base_feature = 64


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.e1_layour = nn.Sequential(
            nn.Conv2d(4, base_feature, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True))
        self.e2_layour = nn.Sequential(
            nn.Conv2d(base_feature, base_feature * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True))
        self.e3_layour = nn.Sequential(
            nn.Conv2d(base_feature * 2, base_feature * 4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True))
        self.e4_layour = nn.Sequential(
            nn.Conv2d(base_feature * 4, base_feature * 8, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True))
        self.e5_layour = nn.Sequential(
            nn.Conv2d(base_feature * 8, base_feature * 8, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True))

        self.d4_layour = nn.Sequential(
            nn.ConvTranspose2d(base_feature * 8, base_feature * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_feature * 8),
            nn.ReLU(True))
        self.d5_layour = nn.Sequential(
            nn.ConvTranspose2d(base_feature * 8, base_feature * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_feature * 4),
            nn.ReLU(True))
        self.d6_layour = nn.Sequential(
            nn.ConvTranspose2d(base_feature * 4, base_feature * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_feature * 2),
            nn.ReLU(True))
        self.d7_layour = nn.Sequential(
            nn.ConvTranspose2d(base_feature * 2, base_feature, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_feature),
            nn.ReLU(True))
        self.d8_layour = nn.Sequential(
            nn.ConvTranspose2d(base_feature, 3, 4, 2, 1, bias=False),
            nn.Tanh())

    def forward(self, x):
        e1 = self.e1_layour(x)
        e2 = self.e2_layour(e1)
        e3 = self.e3_layour(e2)
        e4 = self.e4_layour(e3)
        e5 = self.e5_layour(e4)

        d4 = self.d4_layour(e5)
        d4 = torch.add(d4, e4)
        d5 = self.d5_layour(d4)
        d5 = torch.add(d5, e3)
        d6 = self.d6_layour(d5)
        d6 = torch.add(d6, e2)
        d7 = self.d7_layour(d6)
        d7 = torch.add(d7, e1)
        d8 = self.d8_layour(d7)

        return d8


"""
判别器模型 为单纯的卷积网络
torch.nn.Conv2d(in_channels, out_channels, 
kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
这里备注一下 out_channels
输出通道数代表输出的特征数量，某种意义上对应卷积核的数量，即一个卷积核对应一类特征

这里注意： 
h_out = （h_in + 2 * padding - dilation*（kernel_size - 1） - 1）/ stride  + 1
输出最后两个维度必须是 1 1

（128 + 2 - 3 - 1）/2 + 1 = 64
(64   - 2)/2 + 1 = 32
(32 - 2)/2 + 1   = 16
(16 - 2)/2 + 1   = 8
(8 - 2)/2 + 1    = 4
(4 - 4)/1 + 1    = 1

"""


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Conv2d(7, base_feature, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_feature, base_feature * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_feature * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_feature * 2, base_feature * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_feature * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_feature * 4, base_feature * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_feature * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # nn.Conv2d(base_feature * 8, base_feature * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(base_feature * 8),
            # nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_feature * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.dis(x)
        return x.reshape(x.shape[0], -1)


"""
实例化网络
"""
d_learning_rate = 3e-4  # 3e-4
g_learning_rate = 3e-4
optim_betas = (0.9, 0.999)
criterion = nn.BCELoss()  # 损失函数 - 二进制交叉熵
G = Generator()
D = Discriminator()

g_optimizer = optim.Adam(G.parameters(), lr=d_learning_rate)
d_optimizer = optim.Adam(D.parameters(), lr=d_learning_rate)

"""
预处理
1. 从\CelebA\数据集中选取200张人像图
2. 取180张作为训练集，取20张作为测试集
3. 对训练集的人像进行边缘检测生成边缘图
4. 对训练集的人像进行模糊处理生成噪声图（训练的噪声图应该每轮更新）
5. 对测试集的人像进行边缘检测生成边缘图
6. 对测试集的人像进行模糊处理生成噪声图
"""

ori_file_path = './data/anime-faces/'

"""
数据加载 
"""
transform_3 = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

transform_1 = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.Normalize((0.5), (0.5)), ])


def _get_train_image(batch_size=1):
    """
    同一轮训练伪造图片与真实图片存在对应关系

    1. 从train路径中随机取出 batch_size 张 【边缘图片】
    2. 根据边缘图片的原始图片生成 batch_size 张【噪声图片】
    3. 叠加 【边缘图片】 和 【噪声图片】
    4. 对叠加后的图片进行处理，如下:
        transforms.Resize(image_size)
        transforms.CenterCrop(image_size)
        transforms.ToTensor()
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    """

    noise = []
    ori_result = []
    edge_result = []

    ori_list = os.listdir(ori_file_path)

    # 生成一个随机序列
    numlist = random.sample(range(0, len(ori_list)), batch_size)

    for i in numlist:

        _filename = ori_list[i]
        ori_file = ori_file_path + _filename
        ori_img = cv2.imread(ori_file)

        # generate edge
        img_gray = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
        edge_img = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=7, C=9)

        # generate blur image instead of noise
        ori_img = np.fliplr(ori_img.reshape(-1, 3)).reshape(ori_img.shape)
        for i in range(5):
            randx = randint(0, 205)
            randy = randint(0, 205)
        ori_img[randx:randx + 10, randy:randy + 10] = 255
        blur_img = cv2.blur(ori_img, (100, 100))

        blur_img = transform_3(blur_img)
        noise.append(torch.unsqueeze(blur_img, 0))

        ori_output = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        ori = transform_3(ori_output)
        ori_result.append(torch.unsqueeze(ori, 0))

        edge_img = transform_1(edge_img)
        edge_result.append(torch.unsqueeze(edge_img, 0))

    return torch.cat(ori_result, dim=0), torch.cat(noise, dim=0), torch.cat(edge_result, dim=0)


def to_img(x):
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)
    out = out.view(-1, 3, image_size, image_size)
    return out


"""
CGAN训练逻辑：

1. 获取线条图像 edge_image;
2. 获取模糊图像 blur_image
3. 生成合并图像 combine_image  = edge_image + blur_image
4. 生成图像为   generate_image = G(combine_image)
5. real_to_d = real_image + combine_image
6. fake_to_d = generate_image + combine_image
7. 对判别器优化：real_to_d 判真 fake_to_d 判真
8. fake_to_g = generate_image_1 + combine_image
8. 对生成器优化：fake_to_g 判真

备注：
20200723调整：
20200724调整：cv2 未转rgb

"""


def _show_test_process_data(image):
    # 用以显示tensor的图像数据，测试用
    data = image[0].numpy()
    data = data.transpose((1, 2, 0))
    print(data)
    # data = (data + 1) *0.5 * 255
    print(data)
    plt.imshow(data)


num_epochs = 5000  # 循环次数
for epoch in range(num_epochs):

    # 获取数据
    real_image, noise_image, edge_image = _get_train_image(batch_size)

    # 第一步：训练判别器
    real_label = torch.full((batch_size, 1), 1, dtype=torch.float)
    fake_label = torch.full((batch_size, 1), 0, dtype=torch.float)

    combined_preimage = torch.cat([edge_image, noise_image], dim=1) # [1, 4, 64, 64]

    generate_image = G(combined_preimage) # [1, 3, 64, 64]

    real_image = torch.cat([combined_preimage, real_image], dim=1)  # [1, 7, 64, 64]
    fake_image = torch.cat([combined_preimage, generate_image], dim=1)  # [1, 7, 64, 64]

    # _show_test_process_data(noise_image)

    d_real_decision = D(real_image)
    d_fake_decision = D(fake_image)

    d_real_loss = criterion(d_real_decision, real_label)
    d_fake_loss = criterion(d_fake_decision, fake_label)
    d_loss = d_real_loss + d_fake_loss
    d_optimizer.zero_grad()
    d_loss.backward()
    d_optimizer.step()

    # 第二步：训练生成器
    g_real_decision = D(fake_image.detach())
    g_fake_loss = criterion(g_real_decision, real_label)
    g_optimizer.zero_grad()
    g_fake_loss.backward()
    g_optimizer.step()

    if epoch % 500 == 0 or epoch == 0:
        print("Epoch[{}],g_fake_loss:{:.6f} ,d_loss:{:.6f}"
              .format(epoch, g_fake_loss.data.item(), d_loss.data.item()))
        output = to_img(generate_image)
        save_image(output, './img/cgan/test_' + str(epoch) + '.png')

        test = []
        test_img = cv2.imread('./img/cgan/0000001.jpg')
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        test_pil_img = Image.fromarray(test_img)
        test_input = transform_1(test_pil_img)
        test.append(torch.unsqueeze(test_input, 0))
        data = torch.cat(test, dim=0)
        generate_res = G(Variable(data).cuda())
        save_image(to_img(generate_res), './img/cgan/result.jpg')
