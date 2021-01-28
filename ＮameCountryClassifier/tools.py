import time
import math
import os


def time_since(since):  # 计时模块
	s = time.time() - since
	m = math.floor(s / 60)
	s -= m * 60
	return '%dm%ds' % (m, s)

def create_dir_not_exit(path):  # 如果不存在path文件夹 则创建
	if not os.path.exists(path):
		os.mkdir(path)
