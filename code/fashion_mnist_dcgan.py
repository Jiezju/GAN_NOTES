import os, time
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# G(z)
class generator(nn.Module):
    # initializers
    def __init__(self):
        super(generator, self).__init__()
        self.fc1_1 = nn.Linear(100, 256)
        self.fc1_2 = nn.Linear(10, 256)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 784)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = F.relu(self.fc1_1(input))
        y = F.relu(self.fc1_2(label))
        x = torch.cat([x, y], 1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        x = x.reshape(input.shape[0], 1, 28, 28)

        return x

class discriminator(nn.Module):
    # initializers
    def __init__(self):
        super(discriminator, self).__init__()
        self.fc1 = nn.Conv2d(11, 64, kernel_size=4, stride=2, padding=2)
        self.fc2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=2)
        self.fc3 = nn.Conv2d(128, 128, kernel_size=8, stride=1, padding=0)
        self.fc1_bn = nn.BatchNorm2d(64)
        self.fc2_bn = nn.BatchNorm2d(128)
        self.fc4 = nn.Linear(128, 256)
        self.fc5 = nn.Linear(256, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        label = label.reshape(1, 10, 1, 1) * torch.ones(size=(1, 1, 28, 28), dtype=torch.float)
        x = torch.cat([input, label], 1)
        x = F.leaky_relu(self.fc1_bn(self.fc1(x)), 0.2)
        x = F.leaky_relu(self.fc2_bn(self.fc2(x)), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = x.reshape(-1, 128)
        x = self.fc5(self.fc4(x))
        y = F.sigmoid(x)

        return y


def normal_init(m, mean, std):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


# 定义一个记录loss值的函数，便于绘制loss变化曲线
def show_train_hist(hist, show=False, save=False, path='Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


# training parameters
batch_size = 1
lr = 0.0002
train_epoch = 50

# data_loader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('./data', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True)

# network
G = generator()
D = discriminator()
G.weight_init(mean=0, std=0.02)
D.weight_init(mean=0, std=0.02)
# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# results save folder
if not os.path.isdir('MNIST_cGAN_results'):
    os.mkdir('MNIST_cGAN_results')
if not os.path.isdir('MNIST_cGAN_results/Fixed_results'):
    os.mkdir('MNIST_cGAN_results/Fixed_results')

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

print('training start!')
start_time = time.time()
for epoch in range(train_epoch):
    D_losses = []
    G_losses = []

    # learning rate decay
    if (epoch + 1) == 30:
        G_optimizer.param_groups[0]['lr'] /= 10
        D_optimizer.param_groups[0]['lr'] /= 10
        print("learning rate change!")

    if (epoch + 1) == 40:
        G_optimizer.param_groups[0]['lr'] /= 10
        D_optimizer.param_groups[0]['lr'] /= 10
        print("learning rate change!")

    epoch_start_time = time.time()
    for x_, y_ in train_loader:
        # train discriminator D
        D.zero_grad()

        mini_batch = x_.shape[0]

        # gan label
        real_ = torch.ones(mini_batch, 1)
        fake_ = torch.zeros(mini_batch, 1)

        # one hot label
        y_label_ = torch.zeros(mini_batch, 10)
        y_label_[torch.arange(mini_batch), y_] = 1

        D_result = D(x_, y_label_)
        D_real_loss = BCE_loss(D_result, real_)

        # noise
        z_ = torch.rand((mini_batch, 100))

        G_result = G(z_, y_label_)

        D_result = D(G_result, y_label_)
        D_fake_loss = BCE_loss(D_result, fake_)

        D_train_loss = D_real_loss + D_fake_loss

        D_train_loss.backward()
        D_optimizer.step()

        D_losses.append(D_train_loss.data)

        # train generator G
        G.zero_grad()

        z_ = torch.rand((mini_batch, 100))
        # one hot label
        y_label_ = torch.zeros(mini_batch, 10)
        y_label_[torch.arange(mini_batch), y_] = 1

        G_result = G(z_, y_label_)
        D_result = D(G_result, y_label_)
        G_train_loss = BCE_loss(D_result, real_)
        G_train_loss.backward()
        G_optimizer.step()

        G_losses.append(G_train_loss.data)
        print('G_train_loss:', G_train_loss, 'D_train_loss:', D_train_loss)

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time

    print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % (
    (epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
    torch.mean(torch.FloatTensor(G_losses))))
    fixed_p = 'MNIST_cGAN_results/Fixed_results/MNIST_cGAN_' + str(epoch + 1) + '.png'
    train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)
print("Avg one epoch ptime: %.2f, total %d epochs ptime: %.2f" % (
torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
print("Training finish!... save training results")
