import os
import numpy as np
import tensorflow as tf


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
