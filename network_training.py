import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from dataset import *


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

        )

        self.fc1 = nn.Sequential(
            nn.Linear(8 * 100 * 100, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


# 加载训练数据
train_dataloader = DataLoader(siamese_dataset,
                              shuffle=True,
                              num_workers=8,
                              batch_size=Config.train_batch_size)
# net = SiameseNetwork().cuda()

if __name__ == '__main__':
    net = SiameseNetwork()  # 创建一个Siamese网络的实例
    criterion = ContrastiveLoss()  # 创建对比损失函数的实例
    optimizer = optim.Adam(net.parameters(), lr=0.00000005)  # 创建使用Adam优化器的实例

    counter = []  # 用于记录迭代次数
    loss_history = []  # 用于记录损失值的历史
    iteration_number = 0  # 迭代次数初始化为0

    # 开始训练过程
    for epoch in range(0, Config.train_number_epochs):  # 遍历每个epoch
        for i, data in enumerate(train_dataloader, 0):
            img0, img1, label = data
            optimizer.zero_grad()                       # 清零梯度
            output1, output2 = net(img0, img1)
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()     # 反向传播计算梯度
            optimizer.step()                # 优化器更新模型参数
            if i % 10 == 0:
                print("Epoch number {}\n Current loss {}\n".format(epoch, loss_contrastive.item()))
                iteration_number += 10
                counter.append(iteration_number)                # 记录迭代次数
                loss_history.append(loss_contrastive.item())    # 记录损失值

    # 保存模型
    torch.save(net.state_dict(), f'./facial_matching_model_{Config.train_number_epochs}.pth')

    # 绘制损失曲线
    show_plot(counter, loss_history)
    plt.savefig(f'facial_matching_model_{Config.train_number_epochs}_loss.png')
