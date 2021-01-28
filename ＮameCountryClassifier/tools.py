import time
import math


def time_since(since):  # 计时模块
	s = time.time() - since
	m = math.floor(s / 60)
	s -= m * 60
	return '%dm%ds' % (m, s)
