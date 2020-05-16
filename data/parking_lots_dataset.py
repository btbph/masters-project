import torch
import cv2

from skimage import transform
from torch.utils.data import Dataset
from typing import List


class ParkingLotsDataset(Dataset):
    def __init__(self, image_paths: List[str], classes: List[int], transforms):
        self.image_paths = image_paths
        self.classes = classes
        self.transform = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        image = cv2.imread(self.image_paths[index])
        if image is None:
            raise FileNotFoundError(f'File {self.image_paths[index]} is not found')

        sample = {
            'image': transform.resize(image, (128, 128), anti_aliasing=True, mode='reflect'),
            'image_path': self.image_paths[index],
            'label': self.classes[index]
        }
        if self.transform:
            sample = self.transform(sample)

        return sample
