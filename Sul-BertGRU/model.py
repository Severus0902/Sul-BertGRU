

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#GRU
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers * 1, x.size(0), self.hidden_size).to(device)
        # 前向传播 GRU
        out, _ = self.gru(x, h0)
        return out

class MLP(nn.Module):
    def __init__(self, output_size):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, output_size)
        self.dropout = nn.Dropout(p=0.5)
        # 定义激活函数
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        # 全连接层和激活函数
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

class CNNModel_feature(nn.Module):
    def __init__(self, input_channels, output_size):
        super(CNNModel_feature, self).__init__()
        # 定义卷积层和池化层
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=2, padding=1)  # 输出通道，卷积核3，
        self.pool1 = nn.AvgPool1d(kernel_size=4)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=2, padding=1)
        self.pool2 = nn.AvgPool1d(kernel_size=4)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=2, padding=1)
        self.pool3 = nn.AvgPool1d(kernel_size=4)
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
        return x

def clones(module, N):
    """Product N identical layers."""
    # print("clones!")
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value,mask=None, dropout=None):
    """Compute Scaled Dot Product Attention"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiheadAttention(nn.Module):
    def __init__(self,  d_model,h, dropout=0):
        """Take in model size and number of heads."""
        super(MultiheadAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4) # create 4 linear layers
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)
        # (batch_size, seq_length, d_model)

        # 1) Do all the linear projections in batch from d_model => h * d_k
        query, key, value = [l(x) for l, x in zip(self.linears, (query, key, value))]
        query, key, value = [x.view(batch_size, -1, self.h, self.d_k).transpose(1, 2) for x in (query, key, value)]
        # (batch_size, h, seq_length, d_k)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.linears[-1](x)

class GRU_CNN_Model_p(nn.Module):
    def __init__(self, gru_input_size, gru_hidden_size, gru_num_layers, gru_num_classes,
                 cnn_input_channels, cnn_output_size):
        super(GRU_CNN_Model_p, self).__init__()

        # 定义GRU模型
        self.gru =GRUModel(gru_input_size, gru_hidden_size, gru_num_layers, gru_num_classes)
        self.multiattention1 = MultiheadAttention(gru_hidden_size, 4)
        self.multiattention2 = MultiheadAttention(gru_hidden_size, 4)
        self.multiattention3 = MultiheadAttention(gru_hidden_size, 4)

        # 定义CNN模型
        self.cnn_feature1 = CNNModel_feature(cnn_input_channels, cnn_output_size)
        self.cnn_feature2 = CNNModel_feature(cnn_input_channels, cnn_output_size)

        self.mlp=MLP(cnn_output_size)

    def forward(self, x1, x2, x3):
        # 输入三批数据到LSTM模型提取特征
        gru_out1 = self.gru(x1)
        gru_out2 = self.gru(x2)
        gru_out3 = self.gru(x3)

        attention1=self.multiattention1(gru_out1,gru_out1,gru_out1)
        attention2 = self.multiattention2(gru_out2, gru_out2, gru_out2)
        attention3 = self.multiattention3(gru_out3, gru_out3, gru_out3)

        # 将GRU模型的输出拼接在一起
        att_out = torch.cat((attention1, attention2, attention3), dim=1)
        gru_out = torch.cat((gru_out1, gru_out2, gru_out3), dim=1)

        # 输入GRU模型的输出到CNN模型训练
        att_out = self.cnn_feature1(att_out)
        cnn_out = self.cnn_feature2(gru_out)

        out= torch.cat((att_out, cnn_out), dim=1)
        cnn_out=self.mlp(out)

        # 使用Softmax函数计算每个样本的概率分布
        psx = F.softmax(cnn_out, dim=1)
        return cnn_out,psx
