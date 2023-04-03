import paddle
from paddle.nn import Conv2D, Linear, Embedding
from paddle import to_tensor
import paddle.nn.functional as F

from datasetGenerating import vocab


class RNN(paddle.nn.Layer):
    def __init__(self):
        super(RNN, self).__init__()
        self.dict_dim = vocab["<pad>"]
        self.emb_dim = 128
        self.hid_dim = 128
        self.class_dim = 2
        self.embedding = Embedding(
            self.dict_dim + 1, self.emb_dim,
            sparse=False)
        self._fc1 = Linear(self.emb_dim, self.hid_dim)
        self.lstm = paddle.nn.LSTM(self.hid_dim, self.hid_dim)
        self.fc2 = Linear(19200, self.class_dim)

    def forward(self, inputs):
        # [32, 150]
        emb = self.embedding(inputs)
        # [32, 150, 128]
        fc_1 = self._fc1(emb)#第一层
        # [32, 150, 128]
        x = self.lstm(fc_1)
        x = paddle.reshape(x[0], [0, -1])
        x = self.fc2(x)
        x = paddle.nn.functional.softmax(x)
        return x

if __name__ == '__main__':
    rnn = RNN()
    paddle.summary(rnn, (32, 150), "int64")
