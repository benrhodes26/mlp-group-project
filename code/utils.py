import os
import numpy as np
import tensorflow as tf


def get_learning_rate(current_epoch, init_learning_rate,
                      min_learning_rate, exp_decay, decay_step):
    """At the beginning of training, return init_learning_rate.
    Every decay_step epochs, multiply the learning rate by exp_decay, where
    x equals num_epochs_per_step. Do not lower the learning rate below
    min_learning_rate"""

    current_step = int(current_epoch / decay_step)
    new_learning_rate = init_learning_rate*(exp_decay**current_step)

    return max(min_learning_rate, new_learning_rate)


def events_to_numpy(event_file):
    """Return a numpy array of shape (epochs, metrics)."""
    result = []
    event_triple = []
    print(event_file)
    for event in tf.train.summary_iterator(event_file):
        print(event)
        print(event.summary.value)
        for v in event.summary.value:

            print(v.tag)
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
