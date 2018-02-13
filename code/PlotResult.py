import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pickle

def PlotResult(event_file, epochs, save_dir, train_result, valid_result):
    ''' 
    train_result = []
    valid_result = []
       
    for e in tf.train.summary_iterator(event_file):
        train_val=[]
        valid_val=[]
        for v in e.summary.value:
            print(v.tag, v.simple_value)
            if v.tag == 'train_loss' or v.tag == 'train_accuracy' or v.tag == 'train_auc':
                train_val.append(v.simple_value)
            elif v.tag == 'valid_loss' or v.tag == 'valid_accuracy' or v.tag == 'valid_auc':
                valid_val.append(v.simple_value)
        print(len(train_val))        
        if len(train_val)==3:
            train_result.append(train_val)
        if len(valid_val)==3:
            valid_result.append(valid_val)
    
    train_result = np.array(train_result)
    valid_result = np.array(valid_result)
    '''
    event_file = save_dir+'/train/result.pickle'
    with open(event_file,'wb') as result_file:
        pickle.dump(train_result, result_file)
        pickle.dump(valid_result, result_file)
        
    e = np.arange(0.0, epochs, 1.0)
    print(train_result.shape)
    print(valid_result.shape)
    
    #return train_result
    plt.figure()
    plt.plot(e, train_result[0:epochs, 0], label="train loss")
    plt.plot(e, valid_result[0:epochs, 0], label="valid loss")
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss per epoch')
    plt.savefig(save_dir+'/train/loss.png')

    plt.figure()
    plt.plot(e, train_result[0:epochs, 1], label="train acc")
    plt.plot(e, valid_result[0:epochs, 1], label="valid acc")
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy per epoch')
    plt.savefig(save_dir+'/train/accuracy.png')

    plt.figure()
    plt.plot(e, train_result[0:epochs, 2], label="train auc")
    plt.plot(e, valid_result[0:epochs, 2], label="valid auc")
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('auc')
    plt.title('AUC per epoch')
    plt.savefig(save_dir+'/train/auc.png')