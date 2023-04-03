import numpy as np
import paddle
from matplotlib import pyplot as plt
from paddle.nn import Conv2D, Linear, Embedding
from paddle import to_tensor
import paddle.nn.functional as F

from datasetGenerating import train_loader, test_loader
from RNNModelConfiguration import RNN


def draw_process(title,color,iters,data,label):
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=20)
    plt.ylabel(label, fontsize=20)
    plt.plot(iters, data,color=color,label=label)
    plt.legend()
    plt.grid()
    plt.show()


def train(model):
    model.train()
    opt = paddle.optimizer.Adam(learning_rate=0.002, parameters=model.parameters())

    steps = 0
    Iters, total_loss, total_acc = [], [], []

    for epoch in range(3):
        for batch_id, data in enumerate(train_loader):
            steps += 1
            sent = data[0]
            label = data[1]

            logits = model(sent)
            loss = paddle.nn.functional.cross_entropy(logits, label)
            acc = paddle.metric.accuracy(logits, label)

            if batch_id % 50 == 0:
                Iters.append(steps)
                total_loss.append(loss.numpy()[0])
                total_acc.append(acc.numpy()[0])

                print("epoch: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, loss.numpy()))

            loss.backward()
            opt.step()
            opt.clear_grad()

        # evaluate model after one epoch
        model.eval()
        accuracies = []
        losses = []

        for batch_id, data in enumerate(test_loader):
            sent = data[0]
            label = data[1]

            logits = model(sent)
            loss = paddle.nn.functional.cross_entropy(logits, label)
            acc = paddle.metric.accuracy(logits, label)

            accuracies.append(acc.numpy())
            losses.append(loss.numpy())

        avg_acc, avg_loss = np.mean(accuracies), np.mean(losses)

        print("[validation] accuracy: {}, loss: {}".format(avg_acc, avg_loss))

        model.train()

    paddle.save(model.state_dict(), "model_final.pdparams")

    draw_process("trainning loss", "red", Iters, total_loss, "trainning loss")
    draw_process("trainning acc", "green", Iters, total_acc, "trainning acc")


if __name__ == '__main__':
    model = RNN()
    train(model)


