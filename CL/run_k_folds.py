import os
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import train_test_split
from parameters import get_parameters
import random
from torch.utils.data import Dataset
from model import GRU_CNN_Model
from load_data_p2 import LoadData,split_data
from k_fold import k_fold_train

obj = get_parameters()


def seed_torch(seed=42):
    random.seed(seed)  # Python的随机性
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    # np.random.seed(seed)   # numpy的随机性
    torch.manual_seed(seed)  # torch的CPU随机性，为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # torch的GPU随机性，为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.   torch的GPU随机性，为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True  # 选择确定性算法


def test(left_test_dataset, right_test_dataset, all_test_dataset):
    checkpoint = torch.load('best_model.pth', map_location='cpu')
    model = GRU_CNN_Model(gru_input_size=obj.input_size, gru_hidden_size=obj.hidden_size,
                           gru_num_layers=obj.num_layers,
                           gru_num_classes=obj.num_classes,
                           cnn_input_channels=obj.input_channels, cnn_output_size=obj.output_size).to(device)
    model.load_state_dict(checkpoint)
    y_true = []
    y_pred = []
    total = 0
    correct = 0
    total_loss = 0
    left_test_dataloader = torch.utils.data.DataLoader(left_test_dataset, batch_size=obj.batch_size, shuffle=True,
                                                       generator=torch.Generator().manual_seed(42), drop_last=True)
    right_test_dataloader = torch.utils.data.DataLoader(right_test_dataset, batch_size=obj.batch_size, shuffle=True,
                                                        generator=torch.Generator().manual_seed(42), drop_last=True)
    all_test_dataloader = torch.utils.data.DataLoader(all_test_dataset, batch_size=obj.batch_size, shuffle=True,
                                                      generator=torch.Generator().manual_seed(42), drop_last=True)
    with torch.no_grad():
        for left_test_data, right_test_data, all_test_data in zip(left_test_dataloader, right_test_dataloader,
                                                                  all_test_dataloader):
            y_true_batch = []
            y_pred_batch = []
            left_test_data[0] = left_test_data[0].to(device)
            left_test_data[1] = left_test_data[1].to(device)
            right_test_data[0] = right_test_data[0].to(device)
            right_test_data[1] = right_test_data[1].to(device)
            all_test_data[0] = all_test_data[0].to(device)
            all_test_data[1] = all_test_data[1].to(device)
            left_test_samples, left_test_labels = left_test_data[0], left_test_data[1]
            right_test_samples, right_test_labels = right_test_data[0], right_test_data[1]
            all_test_samples, all_test_labels = all_test_data[0], all_test_data[1]
            # 将三个数据批次作为元组传递给模型进行训练
            test_outputs,_ = model(left_test_samples, right_test_samples, all_test_samples)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(test_outputs, left_test_labels.long())
            _, predicted = torch.max(test_outputs.data, 1)
            y_true_batch += left_test_labels.tolist()
            y_pred_batch += predicted.tolist()
            # 将该批次中的真实标签和预测标签添加到总列表中
            y_true += y_true_batch
            y_pred += y_pred_batch
            total += left_test_labels.size(0)
            _, predicted = torch.max(test_outputs.data, 1)
            correct += (predicted == left_test_labels).sum().item()
            total_loss += loss.item()
        cm = confusion_matrix(y_true, y_pred)  # 计算分类模型的混淆矩阵
        tn, fp, fn, tp = cm.ravel()
        test_mcc = matthews_corrcoef(y_true, y_pred)  # MCC
        test_sn = tp / (tp + fn)  # 敏感性
        test_sp = tn / (tn + fp)  # 特異性
        ac = (tp + tn) / (tp + fp + tn + fn)
        precision = tp / (tp + fp)
        auc = roc_auc_score(y_true, y_pred)
        # 计算训练准确率
        print(total)
        test_accuracy = correct / total
        test_loss = total_loss / (len(left_test_dataloader) + len(right_test_dataloader) + len(all_test_dataloader))
        print(
            'Test Loss: {:.4f}, Test Accuracy: {:.2f}%, MCC: {:.4f}%, SN: {:.4f}%, '
            'SP: {:.4f}% ,AUC: {}, AC:{:.4f}%, PRE:{:.4f}%'.format(
                test_loss, test_accuracy * 100, test_mcc * 100, test_sn * 100, test_sp * 100, auc, ac * 100,
                           precision * 100))


if __name__ == '__main__':
    '''
    分割数据集，按照8：2将数据集划分为训练集和测试集
    '''
    import pickle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)
    seed_torch()
    train_left_dataset, train_right_dataset, train_all_dataset, test_left_dataset, test_right_dataset, test_all_dataset = split_data(device)

    with open('train_left_clean.pkl', 'wb') as f:
        pickle.dump(train_left_dataset, f)

    with open('train_right_clean.pkl', 'wb') as f:
        pickle.dump(train_right_dataset, f)

    with open('train_all_clean.pkl', 'wb') as f:
        pickle.dump(train_all_dataset, f)

    k_fold_train(obj.k_fold, train_left_dataset, train_right_dataset, train_all_dataset)
    print('--------------Test----------------')
    test(test_left_dataset, test_right_dataset, test_all_dataset)
