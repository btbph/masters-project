import os
import torch
from pathlib import Path

DATA_DIR = '/home/m_ulyanov/data/CNRPark'
ANNOTATION_FILE_DIR = os.path.join(os.getcwd(), 'annotation.txt')
CNRPARK_EXTRA_DATA_DIR = os.path.join(str(Path.home()), 'data/CNRPark_extra/PATCHES')

_labels_subdir = 'data/CNRPark_extra/LABELS'
TRAIN_CNRPARK_EXTRA_ANNOTATION = os.path.join(str(Path.home()), _labels_subdir, 'train.txt')
VAL_CNRPARK_EXTRA_ANNOTATION = os.path.join(str(Path.home()), _labels_subdir, 'val.txt')
TEST_CNRPARK_EXTRA_ANNOTATION = os.path.join(str(Path.home()), _labels_subdir, 'test.txt')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
