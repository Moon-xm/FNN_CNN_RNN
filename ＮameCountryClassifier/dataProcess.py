import gzip
import csv
from torch.utils.data import Dataset

train_file = 'dataset/names_train.csv.gz'
test_file = 'dataset/names_test.csv.gz'


class NameDataset(Dataset):
	def __init__(self, train=True):
		filename = train_file if train else test_file
		with gzip.open(filename, 'rt') as f:
			reader = csv.reader(f)
			rows = list(reader)
		self.names = [row[0] for row in rows]
		self.countries = [row[1] for row in rows]
		self.country_ls = list(sorted(set(self.countries)))  # 用set去除重复元素 sorted 按字母排序
		self.country_dic = self.getCountryDic()
		self.country_num = len(self.country_ls)
		self.len = len(self.names)

	def __getitem__(self, index):
		return self.names[index], self.country_dic[self.countries[index]]  # 返回名字字符串及国家对应的索引

	def __len__(self):
		return self.len

	def getCountryDic(self):  # 将列表转换为字典  ls->dic
		country_dic = dict()  # 空字典
		for idx, country_name in enumerate(self.country_ls):
			country_dic[country_name] = idx
		return country_dic

	def idx2country(self, index):  # 根据索引返回对应的国家
		return self.country_ls[index]

	def getCountriesNum(self):  # 返回国家数量
		return self.country_num
