import time
import copy
import torch

from utils import device
from data import train_dataloader, validation_dataloader, COUNT_VAL_SET, COUNT_TEST_SET


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    phases = {
        'train': train_dataloader,
        'val': validation_dataloader
    }
    phases_count = {
        'train': COUNT_TEST_SET,
        'val': COUNT_VAL_SET
    }

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in phases:
            running_loss = 0.0
            running_corrects = 0
            if phase == 'train':
                model.train()
            else:
                model.eval()

            for batch in phases[phase]:
                inputs = batch['image'].to(device, dtype=torch.float)
                labels = batch['label'].to(device, dtype=torch.long)

                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / phases_count[phase]
            epoch_acc = running_corrects.double() / phases_count[phase]
            print(f'{phase} Loss: {epoch_loss} Acc: {epoch_acc}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60}m {time_elapsed % 60}s')
    model.load_state_dict(best_model_wts)
    return model
