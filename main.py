import torch.nn as nn

from torchvision.models import vgg16, resnet18
from torch.optim import SGD, lr_scheduler

from constants import DATA_DIR, ANNOTATION_FILE_DIR
from utils import generate_annotation_file, device
from train import train_model
from constants import CNRPARK_EXTRA_DATA_DIR


def main():
    # TODO: add fixing random state of all variables
    # TODO: add arg parser for generating annotation file
    # TODO: check if dataloader and datasets are working
    # TODO: try to train model with transfer learning
    # generate_annotation_file(DATA_DIR, ANNOTATION_FILE_DIR)
    # model_vgg = vgg16(pretrained=True)
    model = resnet18(progress=True)
    num_filter = model.fc.in_features
    model.classifier = nn.Linear(num_filter, 2)  # 2 is number of classes
    model_vgg = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = SGD(model_vgg.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    best_model = train_model(model_vgg, criterion, optimizer_ft, exp_lr_scheduler)
    print(best_model)


if __name__ == '__main__':
    main()
