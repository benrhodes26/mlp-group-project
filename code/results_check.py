import os 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import get_events_filepath, events_to_numpy

times=['20180309-1303','20180307-0852','20180309-0924']

a=0
losses = []
for i in [0]:
	path = './20180311-0350/'
	a+=1
	train_event = get_events_filepath(path,'train')
	valid_event = get_events_filepath(path,'valid')
	metrics_train = events_to_numpy(train_event)
	metrics_valid = events_to_numpy(valid_event)
	e = np.arange(len(metrics_valid))
	auc = np.array([np.arange(len(metrics_valid)),
				  metrics_train[:len(e),1],
				  metrics_valid[:len(e),1]]).T

	loss = np.array([np.arange(len(metrics_valid)),
				  metrics_train[:len(e),0],
				  metrics_valid[:len(e),0]]).T
	losses.append(metrics_train[:len(e),0])
	losses.append(metrics_valid[:len(e),0])
	print(auc)

	plt.figure()
	train_plt, = plt.plot(e, auc[:,1])
	valid_plt, = plt.plot(e, auc[:,2])
	plt.legend([train_plt, valid_plt], ['train', 'valid'])
	plt.xlabel('Epoch')
	plt.ylabel('AUC')
	plt.title('AUC per epoch')
	plt.savefig(path + '/auc'+str(i)+'.png')

	plt.figure()
	train_plt, = plt.plot(e, loss[:, 1])
	valid_plt, = plt.plot(e, loss[:, 2])
	plt.legend([train_plt, valid_plt], ['train', 'valid'])
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.title('Loss per epoch')
	plt.savefig(path + '/loss' + str(i) + '.png')
'''
l = 0
ll=0
if len(losses[0])<len(losses[2]):
	ll=len(losses[0])
	l = np.arange(len(losses[0]))
else:
	ll=len(losses[2])
	l=np.arange(len(losses[2]))

plt.figure()
train1_plt, = plt.plot(l, losses[0][:ll])
valid1_plt, = plt.plot(l, losses[1][:ll])
train2_plt, = plt.plot(l, losses[2][:ll])
valid2_plt, = plt.plot(l, losses[3][:ll])
plt.legend([train1_plt, valid1_plt,train2_plt, valid2_plt], ['train_fold5', 'valid_fold5','train_fold10', 'valid_fold10'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss per epoch')
plt.savefig(path + '/loss_compare' + '.png')
'''