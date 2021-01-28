### 课堂练习
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# load data
class MyDataset(Dataset):
    def __init__(self,filepath,train=True):
        xy = pd.read_csv(filepath,sep=',')
        self.x = torch.from_numpy(np.array(xy.drop(columns=['id','target'])))
        self.y = torch.LongTensor(np.array(xy['target'].map(lambda x: self.target2id(x))))
        # 划分训练集和测试集 注：加上random_state确保训练集和测试集不重复
        self.x_train, self.x_test, self.y_train, self.y_test =\
            train_test_split(self.x,self.y,test_size=0.2,random_state=0)
        if train is True:
            self.x = self.x_train
            self.y = self.y_train
        else:
            self.x = self.x_test
            self.y = self.y_test
        self.len = len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len

    def target2id(self, target):
        return float(target[-1])-1

train_data = MyDataset(filepath='dataset/otto/train.csv',train=True)
train_loader = DataLoader(train_data,
                          batch_size=32,
                          shuffle=True,
                          num_workers=0)
test_data = MyDataset(filepath='dataset/otto/train.csv',train=False)
test_loader = DataLoader(test_data,
                         batch_size=32,
                         shuffle=False,
                         num_workers=0)

# design model using class
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.l1 = nn.Linear(93, 64)
        self.l2 = nn.Linear(64, 32)
        self.l3 = nn.Linear(32, 16)
        self.l4 = nn.Linear(16, 12)
        self.l5 = nn.Linear(12, 9)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)

model = Net()

# construct loss and optimizer
criterion = nn.CrossEntropyLoss(size_average=True)
optimizer = optim.Adam(model.parameters(),
                       lr=0.01)


# train and test cycle
def train(epoch):
    for batch_id, data in enumerate(train_loader,0):
        x_input, y_input = data
        optimizer.zero_grad()

        # forward + backward + update
        y_pred = model(x_input)
        loss = criterion(y_pred, y_input)
        loss.backward()
        optimizer.step()
    return loss.item()

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            x_input, y_input = data
            y_pred = model(x_input)
            _,predicted = torch.max(y_pred, dim=1)  # 取出行号及对应标签
            total += y_input.size(0)
            correct += (y_input == predicted).sum().item()
            accuracy = (correct/total)
    # print('Accuracy in test data: {:.4f} %'.format((correct/total)*100))

    return accuracy

if __name__ == '__main__':
    loss_ls = []
    accuracy_ls = []
    for epoch in range(300):
        loss = train(epoch)
        accuracy = test()
        loss_ls.append(loss)
        accuracy_ls.append(accuracy)
        accuracy_ls.append(accuracy)
        if (epoch+1) % 10 == 0:
            print('Epoch: {}, Loss: {}, Accuracy: {:.2f}%'.format(epoch+1, loss,accuracy*100))

    plt.plot(loss_ls, label='loss')
    plt.plot(accuracy_ls, label='Accuracy')
    plt.savefig('Accuracy.png')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    plt.show()
