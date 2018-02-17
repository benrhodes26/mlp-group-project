import os
import numpy as np
import tensorflow as tf


def events_to_numpy(event_file):
    """Return a numpy array of shape (epochs, metrics)."""
    result = []
    for event in tf.train.summary_iterator(event_file):
        value_set = []
        is_result = False
        for v in event.summary.value:
            if v.tag == 'loss' or v.tag == 'accuracy' or v.tag == 'auc_1':
                value_set.append(v.simple_value)
                is_result = True
        if is_result:
            result.extend(value_set)
    return np.array(result).reshape(-1,3)


def get_events_filepath(save_dir, name):
    dir_path = os.path.join(save_dir, name)
    file_path = os.path.join(dir_path, os.listdir(dir_path)[0])
    return file_path
