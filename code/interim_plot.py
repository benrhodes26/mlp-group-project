from utils import get_events_filepath
import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf

def events_print(event_file):
	auc = []
	for event in tf.train.summary_iterator(event_file):
		for v in event.summary.value:
			if(v.tag.find('auc_1')>-1):
				#print(v.simple_value)
				auc.append(v.simple_value)
	print(len(auc))
	return auc

experiments_num = [69,70,71,72,78]
dates = ['20180323-0134','20180323-0058','20180323-0125','20180323-0126','20180323-0439']

for i, num in enumerate(experiments_num):
	print('=============Experiment %d=============' %num)
	path = 'experiment'+str(num)+'/'+dates[i]
	events_valid_file = get_events_filepath(path,'valid')
	metrics_valid = events_print(events_valid_file)

	events_train_file = get_events_filepath(path,'train')
	print(events_train_file)
	metrics_train = events_print(events_train_file)
	
	length = len(metrics_train) if len(metrics_train)<len(metrics_valid) else len(metrics_valid)

	e = np.arange(length)

	plt.figure()
	train_plt, = plt.plot(e, metrics_train[:length])
	valid_plt, = plt.plot(e, metrics_valid[:length])
	plt.legend([train_plt, valid_plt], ['train', 'valid'])
	plt.xlabel('Epoch')
	plt.ylabel('AUC')
	plt.title('AUC per epoch')
	plt.savefig(path + 'auc.png')
	
