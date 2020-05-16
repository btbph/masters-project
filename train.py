import time
import copy
import torch

import torch.nn as nn
from torchvision.models import vgg16, resnet18
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from data.transforms import ToTensor

from constants import RANDOM_SEED, LOG_PATH, TRAIN_CNRPARK_EXTRA_ANNOTATION, VAL_CNRPARK_EXTRA_ANNOTATION
from tqdm import tqdm
from utils import device, fix_random_seed, extract_cnrpark_extra_dataset, plot_results
from data.parking_lots_dataset import ParkingLotsDataset
from torchvision.transforms import Compose


def train_model(model, criterion, optimizer, scheduler, X_train, y_train, X_val, y_val, num_epochs=5, save=False, log_path=LOG_PATH, plot_path=None):
    since = time.time()
    count_train_set = len(X_train)
    count_val_set = len(X_val)

    composed = Compose([ToTensor()])

    train_dataloader = DataLoader(ParkingLotsDataset(X_train, y_train, composed),
                                  pin_memory=True,
                                  batch_size=32,
                                  shuffle=True,
                                  num_workers=4)
    validation_dataloader = DataLoader(ParkingLotsDataset(X_val, y_val, composed),
                                       pin_memory=True,
                                       batch_size=32,
                                       shuffle=True,
                                       num_workers=4)

    with open(log_path, 'w'):  # truncate existing file
        pass

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    phases = {
        'train': train_dataloader,
        'val': validation_dataloader
    }
    phases_count = {
        'train': count_train_set,
        'val': count_val_set
    }

    for epoch in tqdm(range(num_epochs)):
        print()
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        log_values = []

        for phase in phases:
            running_loss = 0.0
            running_corrects = 0
            is_train = phase == 'train'

            if is_train:
                model.train()
            else:
                model.eval()

            for batch in phases[phase]:
                inputs = batch['image'].to(device, dtype=torch.float)
                labels = batch['label'].to(device, dtype=torch.long)

                optimizer.zero_grad()
                with torch.set_grad_enabled(is_train):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if is_train:
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
            if is_train:
                scheduler.step()

            epoch_loss = running_loss / phases_count[phase]
            epoch_acc = running_corrects.double() / phases_count[phase]
            log_values.append(str(epoch_loss))
            log_values.append(str(epoch_acc.cpu().numpy()))
            print(f'{phase} Loss: {epoch_loss} Acc: {epoch_acc}')

            if not is_train and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        with open(log_path, 'a') as log_file:
            log_file.write(' '.join(log_values))
            log_file.write('\n')
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60}m {time_elapsed % 60}s')
    model.load_state_dict(best_model_wts)
    if save:
        torch.save(model, './models/last.pth')
    if plot_path is not None:
        plot_results(plot_path)

    return model


# TODO: implement model testing on different metrics [will be implementing in jupyter notebooks]
# TODO: implement cross-validation [need  to be tested]
# TODO: implement correct logging
def main():
    fix_random_seed(RANDOM_SEED)
    model = vgg16()
    model.classifier[6] = nn.Linear(4096, 2)
    # model = resnet18()
    # num_filter = model.fc.in_features
    # model.classifier = nn.Linear(num_filter, 2)  # 2 is number of classes
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    x_train, y_train = extract_cnrpark_extra_dataset(TRAIN_CNRPARK_EXTRA_ANNOTATION)
    x_val, y_val = extract_cnrpark_extra_dataset(VAL_CNRPARK_EXTRA_ANNOTATION)
    best_model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, x_train, y_train, x_val, y_val, num_epochs=2, plot_path='results.png')


if __name__ == '__main__':
    main()
