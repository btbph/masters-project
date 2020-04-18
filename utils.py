import torch
import os

from typing import List, Tuple
from constants import CNRPARK_EXTRA_DATA_DIR

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def extract_from_file(file_name: str) -> Tuple[List[str], List[int]]:
    images = []
    classes = []

    with open(file_name, 'r') as file:
        for line in file:
            split_line = line.split(' ')
            images.append(' '.join(split_line[:-1]))
            classes.append(int(split_line[-1].strip()))

    return images, classes


def generate_annotation_file(root_path: str, res_path: str) -> None:
    with open(res_path, 'w+') as annotation:
        for current_path, dirs, files in os.walk(root_path):
            filtered_files = [file for file in files if not file.startswith('.')]
            if len(filtered_files) != 0 and (current_path.find('busy') != -1 or current_path.find('free') != -1):
                for file in files:
                    file_path = os.path.join(current_path, file)
                    is_busy = current_path.find('busy') != -1
                    annotation.writelines(f'{file_path} {int(is_busy)}\n')


def extract_cnrpark_extra_dataset(file_name: str) -> Tuple[List[str], List[int]]:
    images, classes = extract_from_file(file_name)
    images = [os.path.join(CNRPARK_EXTRA_DATA_DIR, image) for image in images]
    return images, classes
