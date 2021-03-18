import os
from glob import glob
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize, RandomRotation

IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG", ".png",
    ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class BaseAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            Resize(resize, Image.BILINEAR),
            RandomRotation([-8, +8]),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class AugmentationWithGaussianNoise:
    """
    Custom Augmentation Example
    """

    def __init__(self, resize, mean, std, **args):
        noise_mean = args.get("noise_mean", 0.)
        noise_std = args.get("noise_std", 1.)

        self.transform = transforms.Compose([
            Resize(resize, Image.BILINEAR),
            RandomRotation([-8, +8]),
            ToTensor(),
            Normalize(mean=mean, std=std),
            AddGaussianNoise(noise_mean, noise_std)
        ])

    def __call__(self, image):
        return self.transform(image)


class MaskDataset(data.Dataset):
    num_classes = 18

    class MaskLabels:
        mask = 0
        incorrect_mask = 1
        normal = 2

    class GenderLabels:
        male = 0
        female = 1

    class AgeGroup:
        map_label = lambda x: 0 if int(x) < 30 else 1 if int(x) < 60 else 2

    image_paths = []
    labels = []

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        self.image_dir = os.path.join(data_dir, 'images')
        self.df = pd.read_csv(os.path.join(data_dir, 'train.csv'))

        self.mean = mean
        self.std = std
        self.transform = BaseAugmentation((96, 128), mean, std)

        self.setup()
        self.calc_statistics()

    def setup(self):
        for idx, row in self.df.iterrows():
            self.image_paths.extend(glob(os.path.join(self.image_dir, row['path'],
                                                      '*')))  # /mnt/ssd/data/mask/train/images/000001_female_Asian_45/*
        self.image_paths = list(filter(is_image_file, self.image_paths))
        self.labels = [0] * len(self.image_paths)

        for idx in range(len(self.image_paths)):
            image_path = self.image_paths[idx]
            filename = os.path.basename(image_path)
            filename = os.path.splitext(filename)[0]  # mask3
            filename = ''.join([i for i in filename if not i.isdigit()])  # mask
            mask_label = getattr(self.MaskLabels, filename)

            profile = image_path.split('/')[-2]  # 000001_female_Asian_45
            image_id, gender, race, age = profile.split("_")
            gender_label = getattr(self.GenderLabels, gender)
            age_label = self.AgeGroup.map_label(age)
            self.labels[idx] = mask_label * 6 + gender_label * 3 + age_label

    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print(
                "[Warning] Calculating statistics... It can takes huge amounts of time depending on your CPU machine :(")
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        image = self.read_image(index)
        label = self.get_label(index)

        image_transform = self.transform(image)
        return image_transform, label

    def __len__(self):
        return len(self.image_paths)

    def get_label(self, index):
        return self.labels[index]

    def read_image(self, index):
        image_path = self.image_paths[index]
        return Image.open(image_path)


class TestDataset(data.Dataset):
    def __init__(self, img_paths, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        self.img_paths = img_paths
        self.transform = transforms.Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)
