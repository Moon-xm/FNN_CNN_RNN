### 课堂代码
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time
import math

EPOCH = 20
LEARNING_RATE = 0.01
BATCH_SIZE = 64

# design model using class
class InceptionA(nn.Module):  # inception：开端
	def __init__(self, in_channel):
		super(InceptionA, self).__init__()
		self.branch_pool_1 = nn.AvgPool2d(3, padding=1, stride=1)  # 均值池化  padding 保证最后输出的图像W H 和原来的一样
		self.branch_pool_2 = nn.Conv2d(in_channel, 24, kernel_size=1)

		self.branch1x1 = nn.Conv2d(in_channel, 16, kernel_size=1)

		self.branch5x5_1 = nn.Conv2d(in_channel, 16, kernel_size=1)
		self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)

		self.branch3x3_1 = nn.Conv2d(in_channel, 16, kernel_size=1)
		self.branch3x3_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
		self.branch3x3_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

	def forward(self, x):
		branch_pool = self.branch_pool_1(x)
		branch_pool = self.branch_pool_2(branch_pool)

		branch1x1 = self.branch1x1(x)

		branch5x5 = self.branch5x5_1(x)
		branch5x5 = self.branch5x5_2(branch5x5)

		branch3x3 = self.branch3x3_1(x)
		branch3x3 = self.branch3x3_2(branch3x3)
		branch3x3 = self.branch3x3_3(branch3x3)

		outputs = [branch_pool, branch1x1, branch5x5, branch3x3]  # batch_size x channel x W x H
		return torch.cat(outputs, dim=1)

class ResidualBlock(nn.Module):  # 残差网络
	def __init__(self, channel):
		super(ResidualBlock, self).__init__()
		# self.layer1 = nn.Conv2d(channel, channel, kernel_size=1)
		self.layer2 = nn.Sequential(
			nn.BatchNorm2d(channel),
			nn.ReLU(),
			nn.Conv2d(channel, channel, kernel_size=3, padding=1),
			nn.BatchNorm2d(channel),
			nn.ReLU(),
			nn.Conv2d(channel, channel, kernel_size=3, padding=1)
		)

	def forward(self, x):
		# x = self.layer1(x)
		y = self.layer2(x)
		output = F.relu(y + x)
		return output

class MyNet(nn.Module):
	def __init__(self):
		super(MyNet, self).__init__()
		self.layer1 = nn.Sequential(  # 顺序的
			nn.BatchNorm2d(3),  # 归一化层 加上效果会好很多！
			nn.ReLU(),
			nn.Conv2d(3, 10, kernel_size=5),
			nn.MaxPool2d(2)
		)
		self.layer2 = ResidualBlock(10)  # 残差网络
		self.layer3 = InceptionA(10)  # Inception
		self.layer4 = nn.Sequential(
			nn.BatchNorm2d(88),  # 归一化层 加上效果会好很多！
			nn.ReLU(),
			nn.Conv2d(88, 20, kernel_size=5),
			nn.MaxPool2d(2)
		)
		self.layer5 = ResidualBlock(20)
		self.layer6 = InceptionA(20)
		self.fc = nn.Sequential(
			nn.Linear(88*5*5, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, 10)
		)

	def forward(self, x):
		batch_size = x.size(0)
		x = self.layer1(x)
		for i in range(20):
			x = self.layer2(x)
		# x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		for i in range(12):
			x = self.layer5(x)
		# x = self.layer5(x)
		x = self.layer6(x)
		x = x.view(batch_size, -1)
		x = self.fc(x)
		return x

def train():
	for data in train_loader:
		inputs, target = data
		inputs, target = inputs.to(device), target.to(device)
		optimizer.zero_grad()

		# forward + backward + update
		y_pred = model(inputs)
		loss = criterion(y_pred, target)
		loss.backward()
		optimizer.step()
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
			accuracy = correct/total
	# print('Accuracy on test set:{}%:'.format(100*correct/total))
	return accuracy

def time_since(since):  # 计时 返回几分几秒
	s = time.time() - since
	m = math.floor(s / 60)
	s -= m*60
	return '%dm%ds'%(m, s)

# 主函数
if __name__ == '__main__':
	# load data
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.RandomHorizontalFlip(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])

	train_data = datasets.CIFAR10(root='dataset/CIFAR10/',train=True,
								  transform=transform,
								  download=True)
	test_data = datasets.CIFAR10(root='dataset/CIFAR10/',
								 train=False,
								 transform=transform,
								 download=True)
	train_loader = DataLoader(train_data,
							  batch_size=BATCH_SIZE,
							  shuffle=True,
							  num_workers=0)
	test_loader = DataLoader(test_data,
							 batch_size=BATCH_SIZE,
							 shuffle=False,
							 num_workers=0)
	# design model using class
	model = MyNet()  # 实例化model
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	print('Running on: ', device)
	model.to(device)
	print(model)

	# construct loss and optimizer
	criterion = nn.CrossEntropyLoss(reduction='mean')
	optimizer = optim.SGD(model.parameters(),
						  lr=LEARNING_RATE,
						  momentum=0.5)  # 带动量的优化

	# training and test cycle
	start = time.time()
	# loss_ls = []
	accuracy_ls = []
	for epoch in range(EPOCH):
		loss = train()
		accuracy = test()
		total_time = time_since(start)
		print('EPOCH:{}, Loss:{}， accuracy: {:.2f}%, total time: {} '.format(epoch+1, loss, accuracy*100, total_time))
		# loss_ls.append(loss)
		accuracy_ls.append(accuracy)

	epoch_np = np.arange(1, EPOCH+1, 1)
	accuracy_np = np.array(accuracy_ls)
	plt.plot(epoch_np, accuracy_np)
	plt.xlabel('Accuracy')
	plt.ylabel('Loss')
	plt.savefig('AccuracyWithResidualBlockInception.png')
	plt.show()

