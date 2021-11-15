import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, action_dim, device):
        super(DQN, self).__init__()
        #第一个channel是输入的通道数，后面是卷积核的数量，也是输出的通道数
        self.__conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, bias=False)     #建立三个卷积层
        self.__conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.__conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)
        self.__fc1 = nn.Linear(64*7*7, 512)                                     #两个全连接层
        self.__fc2 = nn.Linear(512, action_dim)                                 
        self.__device = device

    def forward(self, x):               
        x = x / 255.
        x = F.relu(self.__conv1(x))                      #激活函数 先卷积再激活
        x = F.relu(self.__conv2(x))
        x = F.relu(self.__conv3(x))
        x = F.relu(self.__fc1(x.view(x.size(0), -1)))    #resize再通过全连接层
        return self.__fc2(x)

    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):               #全连接
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu") #正态分布
            #根据输入计算方差
            module.bias.data.fill_(0.0)                 #偏差置为0
        elif isinstance(module, nn.Conv2d):             #卷积
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
