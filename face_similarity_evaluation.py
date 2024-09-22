from network_training import *
import os
import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F

# 避免某些库版本冲突问题
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
if __name__ == '__main__':
    # 加载模型
    net = SiameseNetwork()
    net.load_state_dict(torch.load(f'./facial_matching_model_{Config.train_number_epochs}.pth'))

    # 创建测试数据集
    folder_dataset_test = dset.ImageFolder(root=Config.testing_dir)
    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                            transform=transforms.Compose([transforms.Resize((100, 100)),
                                                                          transforms.ToTensor()
                                                                          ]),
                                            should_invert=False)
    test_dataloader = DataLoader(siamese_dataset, batch_size=1, shuffle=True)
    dataiter = iter(test_dataloader)
    x0, _, _ = next(dataiter)

    fig, axs = plt.subplots(2, 5)
    # 比较图像对并计算欧式距离
    for i in range(2):
        for j in range(5):  # 每行显示5幅图像
            _, x1, label2 = next(dataiter)
            concatenated = torch.cat((x0, x1), 0)

            output1, output2 = net(x0, x1)  # 使用加载的模型进行前向传播
            euclidean_distance = F.pairwise_distance(output1, output2)  # 计算欧式距离

            # 显示图像和欧式距离
            axs[i, j].imshow(torchvision.utils.make_grid(concatenated).permute(1, 2, 0))
            axs[i, j].set_title('不相似度: {:.2f}'.format(euclidean_distance.item()))
            axs[i, j].axis('off')
    plt.tight_layout()
    plt.show()
