from torchvision.transforms import Compose
from data.parking_lots_dataset import ParkingLotsDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from utils import extract_from_file, extract_cnrpark_extra_dataset
from constants import TRAIN_CNRPARK_EXTRA_ANNOTATION, VAL_CNRPARK_EXTRA_ANNOTATION
from data.transforms import ToTensor

# images, labels = extract_from_file(ANNOTATION_FILE_DIR)
# x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.2)
x_train, y_train = extract_cnrpark_extra_dataset(TRAIN_CNRPARK_EXTRA_ANNOTATION)
x_val, y_val = extract_cnrpark_extra_dataset(VAL_CNRPARK_EXTRA_ANNOTATION)

COUNT_TEST_SET = len(x_train)
COUNT_VAL_SET = len(x_val)

composed = Compose([ToTensor()])

train_dataloader = DataLoader(ParkingLotsDataset(x_train, y_train, composed), batch_size=30, shuffle=True,
                              num_workers=4)
validation_dataloader = DataLoader(ParkingLotsDataset(x_val, y_val, composed), batch_size=30, shuffle=True,
                                   num_workers=4)
