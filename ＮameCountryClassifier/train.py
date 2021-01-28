from dataProcess import NameDataset
from model import GRUModel, USE_GPU, make_tensors
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tools import time_since
import time
import matplotlib.pyplot as plt
import numpy as np

BATCH_SIZE = 64
NUM_WORKERS = 0
HIDDEN_SIZE = 100  # 隐层维度
N_LAYER = 2  # gru层数
N_CHARS = 128  # 字典长度  即所有输入字符的类别 最小是所有大小写字母的数量
LEARNING_RATE = 0.01
EPOCH = 20  # 训练轮数


# device config
if USE_GPU and torch.cuda.is_available():
	device = torch.device('cuda:0')
else:
	device = torch.device('cpu')
print('Running on: ', device)

# load dataset
train_set = NameDataset(train=True)
test_set = NameDataset(train=False)
train_loader = DataLoader(dataset=train_set,
						  batch_size=BATCH_SIZE,
						  shuffle=True,
						  num_workers=NUM_WORKERS)
test_loader = DataLoader(dataset=test_set,
						 batch_size=BATCH_SIZE,
						 shuffle=False,
						 num_workers=NUM_WORKERS)

N_COUNTRY = train_set.getCountriesNum()  # 最终输出维度（类别数）

# load model
model = GRUModel(N_CHARS, HIDDEN_SIZE, N_COUNTRY, N_LAYER)  # 字典长度（嵌入层维度）、 隐层数、 国家数（输出维度）、 GRU层数
model.to(device)

# construct loss and optimizer
criterion = nn.CrossEntropyLoss(reduction='mean')
optimizer = optim.Adam(model.parameters(),
					  lr=LEARNING_RATE)

# define train and test model
def train():
	for (names, countries) in train_loader:
		inputs, seq_lengths, target = make_tensors(names, countries)
		y_pred = model(inputs, seq_lengths)  # forward
		loss = criterion(y_pred, target)
		optimizer.zero_grad()  # 梯度清零
		loss.backward()  # backward
		optimizer.step()  # update
		return loss.item()

def test():
	correct = 0
	total = len(test_set)
	for (name, countries) in test_loader:
		inputs, seq_lengths, target = make_tensors(name, countries)
		output = model(inputs, seq_lengths)
		pred = output.max(dim=1, keepdim=True)[1]
		correct += pred.eq(target.view_as(pred)).sum().item()
		acc = correct / total
		return acc


# train and test cycle
if __name__ == "__main":
	start = time.time()
	acc_ls = []
	# loss_ls = []
	print('='*10,'Start training','='*10)
	for epoch in range(EPOCH):
		loss = train()
		acc = test()
		# loss_ls.append(loss)
		acc_ls.append(acc)
		print('EPOCH: {}, Loss: {:.4f}, Accuracy: {:.2f}%, Time use: {}'.format(
			epoch+1, loss, acc*100, time_since(start)
		))

	# plot the accuracy
	epoch_np = np.arange(1, EPOCH + 1, 1)
	acc_np = np.array(acc_ls)
	plt.plot(epoch_np, acc_np)
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.grid()
	plt.savefig('result/Accuracy.png')
	plt.show()



