import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#GRU模型
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 初始化隐藏状态)
        h0 = torch.zeros(self.num_layers * 1, x.size(0), self.hidden_size).to(device)
        # 前向传播 GRU
        out, _ = self.gru(x, h0)
        return out


# 定义CNN模型
class CNNModel(nn.Module):
    def __init__(self, input_channels, output_size):
        super(CNNModel, self).__init__()

        # 定义卷积层和池化层
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=2, padding=1)  # 输出通道，卷积核3，
        self.pool1 = nn.AvgPool1d(kernel_size=4)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=2, padding=1)
        self.pool2 = nn.AvgPool1d(kernel_size=4)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=2, padding=1)
        self.pool3 = nn.AvgPool1d(kernel_size=4)

        # 定义全连接层
        #self.fc1 = nn.Linear(896, 512)  # !!! before: self.fc1 = nn.Linear(128*6*6, 512)
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, output_size)
        self.dropout = nn.Dropout(p=0.5)
        # 定义激活函数
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 卷积层和池化层
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        x = self.pool3(x)

        # 将张量展平为向量
        x = x.view(x.size(0), -1)

        # 全连接层和激活函数
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x


class GRU_CNN_Model(nn.Module):
    def __init__(self, gru_input_size, gru_hidden_size, gru_num_layers, gru_num_classes,
                 cnn_input_channels, cnn_output_size):
        super(GRU_CNN_Model, self).__init__()

        # 定义GRU模型
        self.gru =GRUModel(gru_input_size, gru_hidden_size, gru_num_layers, gru_num_classes)

        # 定义CNN模型
        self.cnn = CNNModel(cnn_input_channels, cnn_output_size)

    def forward(self, x1, x2, x3):
        # 输入三批数据到GRU模型提取特征
        gru_out1 = self.gru(x1)
        gru_out2 = self.gru(x2)
        gru_out3 = self.gru(x3)

        # 将GRU模型的输出拼接在一起
        gru_out = torch.cat((gru_out1, gru_out2, gru_out3), dim=1)

        # 输入GRU模型的输出到CNN模型训练
        cnn_out = self.cnn(gru_out)

        # 使用Softmax函数计算每个样本的概率分布
        psx = F.softmax(cnn_out, dim=1)
        return cnn_out,psx
