
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, train_left_dataloader, train_right_dataloader, train_all_dataloader, optimizer, criterion):
    correct = 0
    total = 0
    total_loss = 0
    for left_data, right_data, all_data in zip(train_left_dataloader, train_right_dataloader, train_all_dataloader):
        left_data[0] = left_data[0].to(device)
        left_data[1] = left_data[1].to(device)
        right_data[0] = right_data[0].to(device)
        right_data[1] = right_data[1].to(device)
        all_data[0] = all_data[0].to(device)
        all_data[1] = all_data[1].to(device)
        left_samples, left_labels = left_data[0], left_data[1]
        right_samples, right_labels = right_data[0], right_data[1]
        all_samples, all_labels = all_data[0], all_data[1]
        # 将三个数据批次作为元组传递给模型进行训练
        model.train()
        optimizer.zero_grad()
        outputs,_ = model(left_samples, right_samples, all_samples)
        loss = criterion(outputs, left_labels.long())
        loss.backward()
        optimizer.step()
        # 计算训练准确率
        total += left_labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == left_labels).sum().item()
        total_loss += loss.item()
    train_accuracy = correct / total
    avg_loss = total_loss / (len(train_left_dataloader) + len(train_right_dataloader) + len(train_all_dataloader))
    return train_accuracy, avg_loss
