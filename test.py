import torch
import pandas as pd

from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from utils import extract_cnrpark_extra_dataset, fix_random_seed, device, extract_annotation_file, remove_parentheses
from constants import TEST_CNRPARK_EXTRA_ANNOTATION, RANDOM_SEED, PKLOT_DATA_DIR, TEST_PKLOT_ANNOTATION, TRAIN_PKLOT_ANNOTATION

from data.transforms import ToTensor
from data.parking_lots_dataset import ParkingLotsDataset


def test_model(model, X_test, y_test, log_path='./test_data_info.csv'):
    correct = 0
    total = 0
    log_df = pd.DataFrame(columns=['image_path', 'ground_true_label', 'predicted_label'])
    composed = Compose([ToTensor()])

    test_dataloader = DataLoader(
        ParkingLotsDataset(X_test, y_test, composed),
        pin_memory=True,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )

    with torch.no_grad():
        for data in test_dataloader:
            images = data['image'].to(device, dtype=torch.float)
            labels = data['label'].to(device, dtype=torch.float)
            image_paths = data['image_path']

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += torch.sum(predicted.float() == labels.data)

            for image_path, correct_label, predicted_label in zip(image_paths, labels.cpu().numpy(),
                                                                  predicted.cpu().numpy()):
                log_df = log_df.append({
                    'image_path': image_path,
                    'ground_true_label': int(correct_label),
                    'predicted_label': int(predicted_label)
                }, ignore_index=True)
    log_df.to_csv(log_path)
    accuracy = 100 * correct.double() / total
    print(f'Accuracy {accuracy}')


def main():
    # x_test, y_test = extract_cnrpark_extra_dataset(TEST_CNRPARK_EXTRA_ANNOTATION)
    remove_parentheses('/home/m_ulyanov/data/splits/PKLot', ['all.txt', 'train.txt', 'test.txt', 'val.txt'])
    x_test, y_test = extract_annotation_file(TRAIN_PKLOT_ANNOTATION, PKLOT_DATA_DIR)
    print(f'Count of test examples = {len(x_test)}')

    model = torch.load('./models/last.pth')
    print('Model is loaded!')
    model.to(device)
    test_model(model, x_test, y_test)


if __name__ == '__main__':
    fix_random_seed(RANDOM_SEED)
    main()
