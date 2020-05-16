import os
import torch
from pathlib import Path

CNRPARK_DATA_DIR = '/home/m_ulyanov/data/CNRPark'
CNRPARK_EXTRA_DATA_DIR = '/home/m_ulyanov/data/CNRPark_extra/PATCHES'
ANNOTATION_FILE_DIR = os.path.join(os.getcwd(), 'annotation.txt')
CNRPARK_EXTRA_DATA_DIR = os.path.join(str(Path.home()), 'data/CNRPark_extra/PATCHES')

_labels_subdir = 'data/CNRPark_extra/LABELS'
_splits_directory = '/home/m_ulyanov/data/splits'

TRAIN_CNRPARK_EXTRA_ANNOTATION = os.path.join(str(Path.home()), _labels_subdir, 'train.txt')
VAL_CNRPARK_EXTRA_ANNOTATION = os.path.join(str(Path.home()), _labels_subdir, 'val.txt')
TEST_CNRPARK_EXTRA_ANNOTATION = os.path.join(str(Path.home()), _labels_subdir, 'test.txt')
FULL_CNRPARK_ANNOTATION = os.path.join(_splits_directory, 'CNRPark-EXT', 'all.txt')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

RANDOM_SEED = 0
LOG_PATH = './logs/training_logs.txt'

PKLOT_DATA_DIR = '/home/m_ulyanov/data/PKLot/PKLotSegmented'
TRAIN_PKLOT_ANNOTATION = os.path.join(_splits_directory, 'PKLot', 'train.txt')
VAL_PKLOT_ANNOTATION = os.path.join(_splits_directory, 'PKLot', 'val.txt')
TEST_PKLOT_ANNOTATION = os.path.join(_splits_directory, 'PKLot', 'test.txt')
FULL_PKLOT_ANNOTATION = os.path.join(_splits_directory, 'PKLot', 'all.txt')
