import os
import torch
import torch.nn as nn
from parameters import get_parameters
import torch.optim as optim
import random
from torch.utils.data import Dataset, Subset
from model import GRU_CNN_Model_p
from train import train
from val import val

obj = get_parameters()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def k_fold_train(k_fold, dataset_left, dataset_right, dataset_all):
    fold_size = len(dataset_left) // k_fold
    random.seed(42)
    indices = list(range(len(dataset_left)))
    random.shuffle(indices)

    best_acc = 0
    best_model = None
    for i in range(k_fold):
        if not os.path.exists('./Result/fold_{}'.format(i + 1)):
            os.makedirs('./Result/fold_{}'.format(i + 1))
        print('-----------', (i + 1), '---------------')
        val_idx = indices[i * fold_size: (i + 1) * fold_size]
        if i == 0 and k_fold == 1:
            train_idx = indices
        else:
            train_idx = list(set(indices) - set(val_idx))
        left_train_dataset = Subset(dataset_left, train_idx)
        left_val_dataset = Subset(dataset_left, val_idx)
        right_train_dataset = Subset(dataset_right, train_idx)
        right_val_dataset = Subset(dataset_right, val_idx)
        all_train_dataset = Subset(dataset_all, train_idx)
        all_val_dataset = Subset(dataset_all, val_idx)

        left_train_loader = torch.utils.data.DataLoader(left_train_dataset, batch_size=obj.batch_size, shuffle=True,
                                                        generator=torch.Generator().manual_seed(42), drop_last=True)
        left_val_loader = torch.utils.data.DataLoader(left_val_dataset, batch_size=obj.batch_size, shuffle=True,
                                                      generator=torch.Generator().manual_seed(42), drop_last=True)

        right_train_loader = torch.utils.data.DataLoader(right_train_dataset, batch_size=obj.batch_size, shuffle=True,
                                                         generator=torch.Generator().manual_seed(42), drop_last=True)
        right_val_loader = torch.utils.data.DataLoader(right_val_dataset, batch_size=obj.batch_size, shuffle=True,
                                                       generator=torch.Generator().manual_seed(42), drop_last=True)

        all_train_loader = torch.utils.data.DataLoader(all_train_dataset, batch_size=obj.batch_size, shuffle=True,
                                                       generator=torch.Generator().manual_seed(42), drop_last=True)
        all_val_loader = torch.utils.data.DataLoader(all_val_dataset, batch_size=obj.batch_size, shuffle=True,
                                                     generator=torch.Generator().manual_seed(42), drop_last=True)

        model = GRU_CNN_Model_p(gru_input_size=obj.input_size, gru_hidden_size=obj.hidden_size,
                               gru_num_layers=obj.num_layers,
                               gru_num_classes=obj.num_classes,
                               cnn_input_channels=obj.input_channels, cnn_output_size=obj.output_size).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=obj.learning_rate)

        for epoch in range(obj.num_epochs):
            train_accuracy, avg_loss = train(model, left_train_loader, right_train_loader, all_train_loader, optimizer,
                                             criterion)
            print('k_fold [{}/{}], Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(i + 1, k_fold, epoch + 1,
                                                                                          obj.num_epochs, avg_loss,
                                                                                          train_accuracy * 100))
            val_loss, val_accuracy, mcc, tpr, tnr = val(model, left_val_loader,
                                                        right_val_loader,
                                                        all_val_loader, criterion)
            print(
                'k_fold [{}/{}], Epoch [{}/{}], Val Loss: {:.2f}, Val Accuracy: {:.2f}%, MCC: {:.4f}%, TPR: {:.4f}%, '
                'TNR: {:.4f}%'.format(
                    i + 1, k_fold, epoch + 1, obj.num_epochs, val_loss, val_accuracy * 100, mcc * 100, tpr * 100,
                    tnr * 100))
            if val_accuracy > best_acc:
                best_acc = val_accuracy
                best_model = model.state_dict()

    print('best_acc:', best_acc)
    torch.save(best_model, 'best_model_p.pth')


