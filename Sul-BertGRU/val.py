
import torch
from sklearn.metrics import confusion_matrix, matthews_corrcoef

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def val(model, val_loader_left, val_loader_right, val_loader_all, criterion):
    model.eval()
    y_true = []
    y_pred = []
    total = 0
    correct = 0
    total_loss = 0
    with torch.no_grad():
        for left_test_data, right_test_data, all_test_data in zip(val_loader_left, val_loader_right,
                                                                  val_loader_all):
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
            loss = criterion(test_outputs, left_test_labels.long())
            _, predicted = torch.max(test_outputs.data, 1)
            y_true += left_test_labels.tolist()  # 将labels张量中的每个元素加入到列表中
            y_pred += predicted.tolist()
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
        # print(cm)
        tn, fp, fn, tp = cm.ravel()
        mcc = matthews_corrcoef(y_true, y_pred)  # MCC
        tpr = tp / (tp + fn)  # 真正率
        tnr = tn / (tn + fp)  # 真负率
        # 计算训练准确率
        val_accuracy = correct / total
        val_loss = total_loss / (len(val_loader_left) + len(val_loader_right) + len(val_loader_all))
        return val_loss, val_accuracy, mcc, tpr, tnr
