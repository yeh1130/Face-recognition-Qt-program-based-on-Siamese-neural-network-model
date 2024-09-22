import os
import tkinter as tk
from tkinter import filedialog
import cv2
import glob
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from network_training import SiameseNetwork, SiameseNetworkDataset
from PIL import ImageTk, Image
from torchvision.transforms import ToTensor
from torch.utils.data import TensorDataset
from dataset import *


class App:
    def __init__(self, root):
        """初始化应用程序和用户界面."""
        self.root = root
        self.root.title("yjr人脸匹配系统")
        self.root.geometry("800x600")

        # 加载模型
        self.net = SiameseNetwork()
        self.net.load_state_dict(torch.load(f'./facial_matching_model_50.pth'))

        # 创建 GUI 界面
        self.create_widgets()

        self.transform = transforms.Compose([transforms.Resize((100, 100)),
                                             transforms.ToTensor()
                                             ])

    def create_widgets(self):
        """创建界面组件."""
        self.label1 = tk.Label(self.root, text="人脸识别系统", font=("微软雅黑", 16),
                     bg= "#f5f5f5",anchor="center")
        self.label1.grid(row=0, column=0, pady=20, padx=330, sticky="nsew")
        # self.label1.pack(pady=20, padx=330, sticky="nsew")

        self.choose_button = tk.Button(self.root, text="选择文件", command=self.choose_file, font=("Helvetica", 14),
                                       bg="#4CAF50", fg="white", padx=20, pady=10)
        self.choose_button.grid(row=1, column=0, pady=20, padx=330, sticky="nsew")
        # self.choose_button.pack(pady=20, padx=330, sticky="nsew")

        self.result_label = tk.Label(self.root, font=("微软雅黑", 16), anchor="center")
        self.result_label.grid(row=2, column=0, pady=20, padx=330, sticky="nsew")
        # self.result_label.pack(pady=20, padx=330, sticky="nsew")

    def choose_file(self):
        """允许用户选择文件并进行人脸识别."""
        file_path = filedialog.askopenfilename(title="选择文件", filetypes=[("Image Files", ".jpg .jpeg .png .pgm")])

        if file_path:
            # 读取图像文件
            # 将图像转换为 PIL.Image 格式
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            image_toImage = Image.fromarray(image)

            # 显示图像
            self.image_label = tk.Label(self.root)
            self.image_label.grid(row=3, column=0, pady=20, padx=330, sticky="nsew")
            # self.image_label.pack(pady=20, padx=330, sticky="nsew")
            self.image_tk = ImageTk.PhotoImage(image_toImage)
            self.image_label.configure(image=self.image_tk)

            # 获取图像所属的文件夹名
            folder_name = os.path.basename(os.path.dirname(file_path))

            test_face = self.transform(image_toImage)

            # 进行人物识别
            self.result_label.configure(text="正在识别中...", fg="#8B0000")
            self.root.update()  # 更新界面，显示"正在识别中..."


            min_distance, closest_face = self.predict(test_face)

            # 显示结果
            self.result_label.configure(text=f"真实结果：{folder_name[1:]}\n"
                                             f"识别结果：{closest_face[1:]}\n"
                                             f"欧式距离：{min_distance:.2f}")

    def predict(self, test_face):
        x0 = test_face.unsqueeze(0)

        dataset_path = './data/faces/training/'

        dir_list = os.listdir(dataset_path)
        folder_names = list(filter(lambda name: name != 'README', dir_list))


        distance = {}
        for i in folder_names:
            face_folder_name = i
            face_folder_dir = os.path.join(dataset_path, face_folder_name)
            face_folder_list = glob.glob(os.path.join(face_folder_dir, '*'))



            total_distance = 0.0
            for j in range(1, len(face_folder_list) + 1):
                image_name = f"{j}.pgm"
                image_path = os.path.join(face_folder_dir, image_name)

                # face_image = cv2.imread(image_path)
                face_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                # 将图像转换为 tensor 格式

                face_image = Image.fromarray(face_image)
                face_image = self.transform(face_image)

                x1 = face_image.unsqueeze(0)


                """人脸匹配"""
                output1, output2 = self.net(torch.autograd.Variable(x0), torch.autograd.Variable(x1))
                euclidean_distance = F.pairwise_distance(output1, output2)
                total_distance += euclidean_distance.item()

            average_distance = total_distance / len(face_folder_list)
            distance[i] = average_distance

        min_distance = min(distance.values())
        closest_face = min(distance, key=distance.get)

        return min_distance, closest_face


if __name__ == '__main__':
    # 启动 GUI
    root = tk.Tk()
    app = App(root)
    root.mainloop()
