import numpy as np
import paddle

from RNNModelConfiguration import RNN
from dataPreProcessing import ids_to_str
from datasetGenerating import test_loader
from datasetGenerating import vocab

model_state_dict = paddle.load('model_final.pdparams')
model = RNN()
model.set_state_dict(model_state_dict)
model.eval()
label_map = {0 :"是", 1 :"否"}
samples = []
predictions = []
accuracies = []
losses = []

for batch_id, data in enumerate(test_loader):

    sent = data[0]
    label = data[1]

    logits = model(sent)

    for idx ,probs in enumerate(logits):
        # 映射分类label
        label_idx = np.argmax(probs)
        labels = label_map[label_idx]
        predictions.append(labels)
        samples.append(sent[idx].numpy())

    loss = paddle.nn.functional.cross_entropy(logits, label)
    acc = paddle.metric.accuracy(logits, label)

    accuracies.append(acc.numpy())
    losses.append(loss.numpy())

avg_acc, avg_loss = np.mean(accuracies), np.mean(losses)
print("[validation] accuracy: {}, loss: {}".format(avg_acc, avg_loss))
print('数据: {} \n\n是否谣言: {}'.format(ids_to_str(samples[0]), predictions[0]))
