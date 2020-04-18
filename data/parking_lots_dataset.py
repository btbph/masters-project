import torch

from skimage import io, transform
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

        image = io.imread(self.image_paths[index])
        sample = {
            'image': transform.resize(image, (128, 128), anti_aliasing=True, mode='reflect'),
            'label': self.classes[index]
        }
        if self.transform:
            sample = self.transform(sample)

        return sample
