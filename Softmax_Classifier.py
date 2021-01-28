### 课堂代码
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# Load dataset
batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像的维度由W*H*C->C*H*W  W为图片宽度 H为图片高度 C为通道数（channel）
    transforms.Normalize((0.1370, ), (0.3081, ))  # 归一化 两数分别为mean std 前期计算得到
    ])

train_data = datasets.MNIST(root='dataset/MNIST/',
                            train=True,
                            transform=transform,
                            download=True)
train_loader = DataLoader(train_data,
                          batch_size=batch_size,
                          shuffle=True)
test_data = datasets.MNIST(root='dataset/MNIST/',
                           train=False,
                           transform=transform,
                           download=True)
test_loader = DataLoader(test_data,
                         batch_size=batch_size,
                         shuffle=False)

# design model using class
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(784, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 128)
        self.l4 = nn.Linear(128, 64)
        self.l5 = nn.Linear(64, 10)

    def forward(self,x):
        x = x.view(-1, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)

model = Net()  # 实例化model

# construct loss and optimizer
criterion = nn.CrossEntropyLoss(size_average=True)
optimizer = optim.SGD(model.parameters(),
                      lr=0.01,
                      momentum=0.5)  # 带动量的优化

# training and test cycle
def train(epoch):
    running_loss = 0.0
    for batch_id, data in enumerate(train_loader,0):
        inputs, target = data
        optimizer.zero_grad()

        # forward + backward + update
        y_pred = model(inputs)
        loss = criterion(y_pred, target)
        loss.backward()
        optimizer.step()

        # running_loss += loss.item()  # 一定要记得用item()否则会构建计算图
        if (batch_id+1)%100 == 0:
            print('\tepoch:{}, batch_id:{}, Loss:{}'.format(epoch+1, batch_id+1, loss))
            # running_loss = 0.0
    return loss


def test():  # 注意这里无epoch参数
    correct = 0
    total = 0
    with torch.no_grad():  # 测试部分不计算梯度
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)  # 取出行号和对应最大值下标
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy on test set:{}%:'.format(100*correct/total))

# 主函数
if __name__ == '__main__':
    loss_ls = []
    for epoch in range(10):
        loss = train(epoch)
        if (epoch+1)%1 == 0:
            print('EPOCH:{}, Loss:{}'.format(epoch+1, loss))
        loss_ls.append(loss)
        test()

    plt.plot(loss_ls)
    plt.xlabel('EPOCH')
    plt.ylabel('Loss')
    plt.show()

