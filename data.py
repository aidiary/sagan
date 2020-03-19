from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms


def make_datapath_list():
    train_img_list = list()

    for img_idx in range(200):
        img_path = "./data/img_78/img_7_" + str(img_idx) + '.jpg'
        train_img_list.append(img_path)

        img_path = "./data/img_78/img_8_" + str(img_idx) + '.jpg'
        train_img_list.append(img_path)

    return train_img_list


class ImageTransform():
    def __init__(self, mean, std):
        self.data_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)])

    def __call__(self, img):
        return self.data_transform(img)


class GAN_Img_Dataset(data.Dataset):
    def __init__(self, file_list, transform):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path)

        img_transformed = self.transform(img)

        return img_transformed
