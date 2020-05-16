import torch.nn as nn
from torchvision.models import vgg16, resnet18, resnet50
from torch.optim import SGD, lr_scheduler

from constants import (
    TRAIN_CNRPARK_EXTRA_ANNOTATION,
    VAL_CNRPARK_EXTRA_ANNOTATION,
    CNRPARK_EXTRA_DATA_DIR,

    TRAIN_PKLOT_ANNOTATION,
    VAL_PKLOT_ANNOTATION,
    PKLOT_DATA_DIR,
    RANDOM_SEED
)
from itertools import permutations
from utils import fix_random_seed, device, extract_annotation_file, remove_parentheses
from train import train_model
from test import test_model


def main():
    fix_random_seed(RANDOM_SEED)
    models = [('vgg', vgg16()), ('resnet18', resnet18()), ('resnet50', resnet50())]
    remove_parentheses('/home/m_ulyanov/data/splits/PKLot', ['all.txt', 'train.txt', 'test.txt', 'val.txt'])

    for model_name, model in models:
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer_ft = SGD(model.parameters(), lr=0.001, momentum=0.9)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

        # train annotation file, val annotation file, full annotation file, base data dir
        cross_val_sets = [
            (TRAIN_CNRPARK_EXTRA_ANNOTATION, VAL_CNRPARK_EXTRA_ANNOTATION, TRAIN_CNRPARK_EXTRA_ANNOTATION, CNRPARK_EXTRA_DATA_DIR),
            (TRAIN_PKLOT_ANNOTATION, VAL_PKLOT_ANNOTATION, TRAIN_PKLOT_ANNOTATION, PKLOT_DATA_DIR)
        ]
        for index, (first_set, second_set) in enumerate(permutations(cross_val_sets)):
            x_train, y_train = extract_annotation_file(first_set[0], first_set[-1])
            x_val, y_val = extract_annotation_file(first_set[1], first_set[-1])

            plot_name = f'./logs/{model_name}_{index}.png'
            logs_path = f'./logs/{model_name}_{index}.txt'
            trained_model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, x_train, y_train, x_val, y_val, plot_path=plot_name, save=False, num_epochs=7, log_path=logs_path)

            x_test, y_test = extract_annotation_file(second_set[2], second_set[-1])
            test_log_path = f'./logs/{model_name}_{index}.csv'
            test_model(trained_model, x_test, y_test, log_path=test_log_path)


if __name__ == '__main__':
    main()
