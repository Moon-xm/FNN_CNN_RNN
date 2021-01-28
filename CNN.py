### 课堂代码
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
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
                          shuffle=True,
                          num_workers=0)
test_data = datasets.MNIST(root='dataset/MNIST/',
                           train=False,
                           transform=transform,
                           download=True)
test_loader = DataLoader(test_data,
                         batch_size=batch_size,
                         shuffle=False,
                         num_workers=0)

# design model using class
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(10,20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(320, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self,x):
        batch_size = x.size(0)  # 等价于x.shape[0]
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x

model = Net()  # 实例化model
device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')
print('Running on: ',device)
model.to(device)

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
        inputs, target = inputs.to(device), target.to(device)
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
            images, labels = images.to(device), labels.to(device)
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

