import os
import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt


def get_learning_rate(current_epoch, init_learning_rate,
                      min_learning_rate, exp_decay, decay_step):
    """At the beginning of training, return init_learning_rate.
    Every decay_step epochs, multiply the learning rate by exp_decay, where
    x equals num_epochs_per_step. Do not lower the learning rate below
    min_learning_rate"""

    current_step = int(current_epoch / decay_step)
    new_learning_rate = init_learning_rate*(exp_decay**current_step)

    return max(min_learning_rate, new_learning_rate)


def log_learning_rate_and_grad_norms(sess, model, inputs, targets, target_ids,
                                     learning_rate, keep_prob):
    # optional logging for debugging.
    print("learning rate is: {}".format(learning_rate))
    for gv in model.grads_and_vars:
        _ = sess.run([tf.Print(gv, [tf.norm(gv[0]), gv[1].name],
                               message="Grad norm is: ")],
                     feed_dict={model.inputs: inputs,
                                model.targets: targets,
                                model.target_ids: target_ids,
                                model.learning_rate: learning_rate,
                                model.keep_prob: float(keep_prob)})
        

def events_to_numpy(event_file):
    """Return a numpy array of shape (epochs, metrics)."""
    result = []
    event_triple = []
    for event in tf.train.summary_iterator(event_file):
        for v in event.summary.value:
            if v.tag == 'loss' or v.tag == 'accuracy' or v.tag == 'auc_1':

                event_triple.append(v.simple_value)

        if len(event_triple) == 3:
            # only extend the result vector if we have loss, acc and auc, since
            # it's possible training gets interrupted after only getting loss
            result.extend(event_triple)
            event_triple = []

    return np.array(result).reshape(-1, 3)


def get_events_filepath(save_dir, name):
    dir_path = os.path.join(save_dir, name)
    file_path = os.path.join(dir_path, os.listdir(dir_path)[0])
    return file_path


def plot_learning_curves(save_dir, num_epochs):
    # Get and save loss, accuracy, and auc metrics
    events_file_train = get_events_filepath(save_dir, 'train')
    metrics_train = events_to_numpy(events_file_train)
    np.save(os.path.join(save_dir, 'metrics_train'), metrics_train)

    events_file_valid = get_events_filepath(save_dir, 'valid')
    metrics_valid = events_to_numpy(events_file_valid)
    np.save(os.path.join(save_dir, 'metrics_valid'), metrics_valid)

    # plot metrics
    e = np.arange(num_epochs)
    plt.figure()
    train_plt, = plt.plot(e, metrics_train[:, 0])
    valid_plt, = plt.plot(e, metrics_valid[:, 0])
    plt.legend([train_plt, valid_plt], ['train', 'valid'])
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.title('Loss per epoch')
    plt.savefig(save_dir + '/loss.png')

    plt.figure()
    train_plt, = plt.plot(e, metrics_train[:, 1])
    valid_plt, = plt.plot(e, metrics_valid[:, 1])
    plt.legend([train_plt, valid_plt], ['train', 'valid'])
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('AUC per epoch')
    plt.savefig(save_dir + '/auc.png')

    plt.figure()
    train_plt, = plt.plot(e, metrics_train[:, 2])
    valid_plt, = plt.plot(e, metrics_valid[:, 2])
    plt.legend([train_plt, valid_plt], ['train', 'valid'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per epoch')
    plt.savefig(save_dir + '/accuracy.png')

