from torchvision import datasets, transforms
from base import BaseDataLoader
import os
from pathlib import Path
from skimage import io
import torch
from skimage.transform import resize
import numpy as np
from Cutout import Cutout
import copy

class DogBreedDataset:
    def __init__(self, data_path: str):
        self.data, self.targets = self._load_data(data_path)

        self.trsfm = transforms.Compose([
          transforms.ToTensor(),
          transforms.RandomRotation(30),
          transforms.RandomResizedCrop(224),
          transforms.RandomHorizontalFlip(),
          # transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)),
          transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
          # Cutout(1, 64)
          ])

        self.classes = { }

        for idx, filename in enumerate(os.listdir(Path(data_path))):
          self.classes[filename] = idx

    def __getitem__(self, index):
        img = io.imread(self.data[index])
        img = torch.from_numpy(img).float()
        img_resize = resize(img, (224, 224))
        # if self.transform == True:
        img = self.trsfm(img_resize)

        label = self.targets[index]
        label = torch.tensor(self.classes[label])

        return img, label

    @staticmethod
    def _load_data(path: str):
        data = []
        targets = []

        for i in Path(path).glob("**/*.jpg"):
            data.append(i)
            targets.append(str(i).split('/')[-2])
        return data, targets

    def __len__(self):
        return len(self.data)

    def validation_transform(self):
      self.trsfm = transforms.Compose([
          transforms.ToTensor(),
          transforms.Resize(255),
          transforms.CenterCrop(224),
          transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
      ])

class DogBreedDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        self.dataset = DogBreedDataset(data_path=self.data_dir)

        if training is False:
          test_dataset = copy.copy(self.dataset)
          test_dataset.validation_transform()
          self.dataset = test_dataset

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        # training을 인자값으로 넘겨줄 경우, 에러가 발생함
