import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

train_dir = "20180211-1522/train/events.out.tfevents.1518362537.ashbury.inf.ed.ac.uk"

loss = []
acc = []
auc = []

for e in tf.train.summary_iterator(train_dir):

		for v in e.summary.value:
				if v.tag=='loss':
						loss.append(float(v.simple_value))
				elif v.tag=='accuracy_1':
						acc.append(v.simple_value)
				elif v.tag=='auc_1':
						auc.append(v.simple_value)

print(loss)
epoch = np.arange(0.0, 5.0, 1.0)
plt.plot(epoch, loss,'r', acc,'g', auc,'b')
plt.savefig("fig.png")
