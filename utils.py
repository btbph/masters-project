import torch
import os
import numpy as np
import random
import matplotlib.pyplot as plt

from typing import List, Tuple
from constants import CNRPARK_EXTRA_DATA_DIR, LOG_PATH

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
    return extract_annotation_file(file_name, CNRPARK_EXTRA_DATA_DIR)


def extract_annotation_file(file_path: str, base_file_directory: str) -> Tuple[List[str], List[int]]:
    images, classes = extract_from_file(file_path)
    images = [os.path.join(base_file_directory, image) for image in images]
    return images, classes


def plot_results(save_path: str):
    titles = ['Loss', 'Accuracy', 'Validation loss', 'Validation accuracy']
    loss = []
    val_loss = []
    accuracy = []
    val_accuracy = []

    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    with open(LOG_PATH, 'r') as log_file:
        for line in log_file:
            results = list(map(float, line.split(' ')))
            loss.append(results[0])
            accuracy.append(results[1])
            val_loss.append(results[2])
            val_accuracy.append(results[3])

    data = [loss, accuracy, val_loss, val_accuracy]
    for i in range(4):
        ax[i].plot(range(1, len(data[i]) + 1), data[i], linewidth=2, marker='.', markersize=8)
        ax[i].set_title(titles[i])
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)


def fix_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def remove_parentheses(base_dir: str, files: List[str]) -> None:
    tmp_file_name = 'tmp.txt'
    for file in files:
        with open(os.path.join(base_dir, file), 'r') as input_file, open(tmp_file_name, 'w') as new_file:
            for line in input_file:
                if '(2)' in line:
                    file_path, label = line.split(' ')
                    file_name, extension = file_path.split('.')
                    file_name = file_name[:-3]
                    new_file.write(f'{file_name}.{extension} {label}')
                else:
                    new_file.write(line)
        os.remove(os.path.join(base_dir, file))
        os.rename(tmp_file_name, os.path.join(base_dir, file))
