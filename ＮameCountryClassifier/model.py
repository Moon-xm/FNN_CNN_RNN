import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence

USE_GPU = False


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, bidirectional=True):
        super(GRUModel, self).__init__()
        self.embedding_dim = hidden_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = 2 if bidirectional else 1
        self.embedding = nn.Embedding(input_size, self.embedding_dim)
        self.gru = nn.GRU(self.embedding_dim, hidden_size, num_layers,
                          bidirectional=bidirectional)  # 双向
        self.fc = nn.Linear(hidden_size * self.bidirectional, output_size)

    def forward(self, x, seq_lengths):
        # x shape: BxS -> SxB  S为max_seq_len B为batch_size
        x = x.t()  # [64, 12]->[12, 64]
        batch_size = x.size(1)  # B = 64

        hidden = self._init_hidden(batch_size)  # hidden shape see _init_hidden  [4, 64, 100]
        embedding = self.embedding(x)  # embedding shape: max_seq_len x batch_size x embedding_dim  [12, 64, 100]

        # pick them up
        gru_input = pack_padded_sequence(embedding, seq_lengths)  # 将序列单元按长度降序排序  [[448, 100], 12]
        output, hidden = self.gru(gru_input, hidden)
        if self.bidirectional == 2:
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)  # 双向GRU hidden有两项输出 在B维度进行拼接
        else:
            hidden_cat = hidden[-1]

        out = self.fc(hidden_cat)
        return out

    def _init_hidden(self, batch_size):  # 单下划线表示只能在class内部使用（只是一种约定）
        hidden = torch.zeros(self.num_layers * self.bidirectional,
                             batch_size,
                             self.hidden_size)
        return create_tensor(hidden)


def make_tensors(names, countries):  # 将名字转化为Tensor
    sequences_and_lengths = [name2ls(name) for name in names]
    name_sequences = [s1[0] for s1 in sequences_and_lengths]
    seq_lengths = torch.LongTensor([s1[1] for s1 in sequences_and_lengths])
    countries = countries.long()

    # 将名字转化为Tensor, [batch_size, seq_len]
    seq_tensor = torch.zeros(len(name_sequences), seq_lengths.max()).long()  # 先置为全0 再填充非零部分
    for idx, (seq, seq_len) in enumerate(zip(name_sequences, seq_lengths)):
        seq_tensor[idx, : seq_len] = torch.LongTensor(seq)

    # 按序列长度降序排序
    seq_lengths, perm_idx = seq_lengths.sort(dim=0, descending=True)
    seq_tensor = seq_tensor[perm_idx]
    countries = countries[perm_idx]

    return create_tensor(seq_tensor), \
           create_tensor(seq_lengths), \
           create_tensor(countries)


def name2ls(name):  # 将每个字符转化为ASCLL存储再列表中 返回ASCLL列表及列表长度
    arr = [ord(c) for c in name]
    return arr, len(arr)


def create_tensor(tensor):  # 将tensor转移到gpu上运行（如果设置可用）
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda:0')
        tensor = tensor.to(device)
    return tensor
