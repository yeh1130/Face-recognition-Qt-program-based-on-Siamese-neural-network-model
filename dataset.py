import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps

"""将Tensor图像显示在matplotlib窗口"""
def imshow(img, text=None, should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()


class Config:
    training_dir = "./data/faces/training/"
    testing_dir = "./data/faces/testing/"
    train_batch_size = 64
    train_number_epochs = 50

# 数据集类定义
class SiameseNetworkDataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        """初始化孪生网络数据集"""
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert
    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        """确保50%的概率选择同类或不同类的图像"""
        should_get_same_class = random.randint(0, 1)
        while True:
            img1_tuple = random.choice(self.imageFolderDataset.imgs)
            if (img0_tuple[1] == img1_tuple[1]) == should_get_same_class:
                break

        img0 = self.load_image(img0_tuple[0])
        img1 = self.load_image(img1_tuple[0])

        label = torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))
        return img0, img1, label

    def load_image(self, image_path):
        """加载并处理图像"""
        img = Image.open(image_path).convert("L")
        if self.should_invert:
            img = ImageOps.invert(img)
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imageFolderDataset.imgs)

# 创建ImageFolder对象
folder_dataset = dset.ImageFolder(root=Config.training_dir)

# 创建SiameseNetworkDataset对象
siamese_dataset = SiameseNetworkDataset(
    imageFolderDataset=folder_dataset,
    transform=transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor()
    ]),
    should_invert=False
)

# 数据可视化查看
if __name__ == '__main__':
    vis_dataloader = DataLoader(siamese_dataset, shuffle=True, num_workers=8, batch_size=8)
    dataiter = iter(vis_dataloader)
    example_batch = next(dataiter)
    concatenated = torch.cat((example_batch[0], example_batch[1]), 0)
    print(example_batch[2].numpy())
    imshow(torchvision.utils.make_grid(concatenated))
