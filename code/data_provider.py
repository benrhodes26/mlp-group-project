# -*- coding: utf-8 -*-
"""Data providers.

This module provides classes for loading datasets and iterating over batches of
data points.
"""

import numpy as np
import os
import scipy.sparse as sp

from sklearn.model_selection import KFold
DEFAULT_SEED = 22012018


class DataProvider(object):
    """Generic data provider."""

    def __init__(self, inputs, targets, batch_size, max_num_batches=-1,
                 shuffle_order=True, rng=None):
        """Create a new data provider object.

        Args:
            inputs (ndarray): Array of data input features of shape
                (num_data, input_dim).
            targets (ndarray): Array of data output targets of shape
                (num_data, output_dim) or (num_data,) if output_dim == 1.
            batch_size (int): Number of data points to include in each batch.
            max_num_batches (int): Maximum number of batches to iterate over
                in an epoch. If `max_num_batches * batch_size > num_data` then
                only as many batches as the data can be split into will be
                used. If set to -1 all of the data will be used.
            shuffle_order (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
        """
        self.inputs = inputs
        self.targets = targets
        if batch_size < 1:
            raise ValueError('batch_size must be >= 1')
        self._batch_size = batch_size
        if max_num_batches == 0 or max_num_batches < -1:
            raise ValueError('max_num_batches must be -1 or > 0')
        self._max_num_batches = max_num_batches
        self._update_num_batches()
        self.shuffle_order = shuffle_order
        self._current_order = np.arange(inputs.shape[0])
        if rng is None:
            rng = np.random.RandomState(DEFAULT_SEED)
        self.rng = rng
        self.new_epoch()

    @property
    def batch_size(self):
        """Number of data points to include in each batch."""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        if value < 1:
            raise ValueError('batch_size must be >= 1')
        self._batch_size = value
        self._update_num_batches()

    @property
    def max_num_batches(self):
        """Maximum number of batches to iterate over in an epoch."""
        return self._max_num_batches

    @max_num_batches.setter
    def max_num_batches(self, value):
        if value == 0 or value < -1:
            raise ValueError('max_num_batches must be -1 or > 0')
        self._max_num_batches = value
        self._update_num_batches()

    def _update_num_batches(self):
        """Updates number of batches to iterate over."""
        # maximum possible number of batches is equal to number of whole times
        # batch_size divides in to the number of data points which can be
        # found using integer division
        possible_num_batches = self.inputs.shape[0] // self.batch_size
        if self.max_num_batches == -1:
            self.num_batches = possible_num_batches
        else:
            self.num_batches = min(self.max_num_batches, possible_num_batches)

    def __iter__(self):
        """Implements Python iterator interface.

        This should return an object implementing a `next` method which steps
        through a sequence returning one element at a time and raising
        `StopIteration` when at the end of the sequence. Here the object
        returned is the DataProvider itself.
        """
        return self

    def new_epoch(self):
        """Starts a new epoch (pass through data), possibly shuffling first."""
        self._curr_batch = 0
        if self.shuffle_order:
            self.shuffle()

    def __next__(self):
        return self.next()

    def reset(self):
        """Resets the provider to the initial state."""
        inv_perm = np.argsort(self._current_order)
        self._current_order = self._current_order[inv_perm]
        self.inputs = self.inputs[inv_perm]
        self.targets = self.targets[inv_perm]
        self.new_epoch()

    def shuffle(self):
        """Randomly shuffles order of data."""
        perm = self.rng.permutation(self.inputs.shape[0])
        self._current_order = self._current_order[perm]
        self.inputs = self.inputs[perm]
        self.targets = self.targets[perm]

    def next(self):
        """Returns next data batch or raises `StopIteration` if at end."""
        if self._curr_batch + 1 > self.num_batches:
            # no more batches in current iteration through data set so start
            # new epoch ready for another pass and indicate iteration is at end
            self.new_epoch()
            raise StopIteration()
        # create an index slice corresponding to current batch number
        batch_slice = slice(self._curr_batch * self.batch_size,
                            (self._curr_batch + 1) * self.batch_size)
        inputs_batch = self.inputs[batch_slice]
        targets_batch = self.targets[batch_slice]
        self._curr_batch += 1
        return inputs_batch, targets_batch


class ASSISTDataProvider(DataProvider):
    """Data provider for ASSISTments 2009/2015 student assessment data set."""

    def __init__(self, data_dir, which_set='train', which_year='09', batch_size=100,
                 max_num_batches=-1, shuffle_order=True, rng=None, data=None):
        """Create a new ASSISTments data provider object.

        Args:
            which_set: One of 'train' or 'test'. Determines which
                portion of the ASSIST data this object should provide.
            which_year: either '09' or '15'. Determines which dataset to use.
            batch_size (int): Number of data points to include in each batch.
            max_num_batches (int): Maximum number of batches to iterate over
                in an epoch. If `max_num_batches * batch_size > num_data` then
                only as many batches as the data can be split into will be
                used. If set to -1 all of the data will be used.
            shuffle_order (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
            data: (inputs, target): if not None, use this data instead of
                loading from file
        """
        expanded_data_dir = os.path.expanduser(data_dir)
        data_path = os.path.join(expanded_data_dir, 'assist{0}-{1}'.format(which_year, which_set))
        self._validate_inputs(which_set, which_year, data_path)
        self.which_set = which_set
        self.which_year = which_year
        self.data_dir = data_dir
        self.num_classes = 2

        if data:
            inputs, targets, self.target_ids = data['inputs'], data['targets'], data['target_ids']
            self.max_num_ans, self.max_prob_set_id = data['max_num_ans'], data['max_prob_set_id']
        else:
            inputs = sp.load_npz(data_path + '-inputs.npz')
            loaded = np.load(data_path + '-targets.npz')
            targets, self.target_ids = loaded['targets'],  loaded['target_ids']
            self.max_num_ans, self.max_prob_set_id = int(loaded['max_num_ans']), int(loaded['max_prob_set_id'])
        self.encoding_dim = 2 * self.max_prob_set_id + 1

        # pass the loaded data to the parent class __init__
        super(ASSISTDataProvider, self).__init__(
            inputs, targets, batch_size, max_num_batches, shuffle_order, rng)

    def next(self):
        """Returns next data batch or raises `StopIteration` if at end."""
        if self._curr_batch + 1 > self.num_batches:
            # no more batches in current iteration through data set so start
            # new epoch ready for another pass and indicate iteration is at end
            self.new_epoch()
            raise StopIteration()
        # create an index slice corresponding to current batch number
        batch_slice = slice(self._curr_batch * self.batch_size,
                            (self._curr_batch + 1) * self.batch_size)
        inputs_batch = self.inputs[batch_slice]
        targets_batch = self.targets[batch_slice]
        target_ids_global = self.target_ids[batch_slice]
        self._curr_batch += 1

        # extract one-hot encoded feature vectors and reshape/regroup them
        # so we can feed them to the RNN
        batch_inputs = self._extract_rnn_inputs(inputs_batch)

        # targets_batch is a list of lists, which we need to flatten
        batch_targets = [i for sublist in targets_batch for i in sublist]

        # during learning, the data for each student in a batch gets shuffled together.
        # hence, we need a vector of indices to locate their predictions after learning
        a = self.max_num_ans * self.max_prob_set_id
        batch_target_ids = [a*i + np.array(target_ids_global[i]) for i in range(self.batch_size)]
        batch_target_ids = [i for sublist in batch_target_ids for i in sublist]

        return batch_inputs, batch_targets, batch_target_ids

    def reset(self):
        """Resets the provider to the initial state."""
        inv_perm = np.argsort(self._current_order)
        self._current_order = self._current_order[inv_perm]
        self.inputs = self.inputs[inv_perm]
        self.targets = self.targets[inv_perm]
        self.target_ids = self.target_ids[inv_perm]
        self.new_epoch()

    def shuffle(self):
        """Randomly shuffles order of data."""
        perm = self.rng.permutation(self.inputs.shape[0])
        self._current_order = self._current_order[perm]
        self.inputs = self.inputs[perm]
        self.targets = self.targets[perm]
        self.target_ids = self.target_ids[perm  ]

    def _extract_rnn_inputs(self, inputs_batch):
        x = inputs_batch.toarray()
        x = x.reshape(-1, self.max_num_ans, self.encoding_dim)
        x = x.transpose([1, 0, 2])  # sort by ordered answered, and then by student
        x.reshape(-1, self.encoding_dim)
        x = np.split(x, self.max_num_ans, axis=0)
        return x

    def get_k_folds(self, k):
        """ Returns k pairs of DataProviders: (train_data_provider, val_data_provider)
        where the data split in each tuple is determined by k-fold cross val."""

        assert self.which_set == 'train', (
            'Expected which_set to be train. '
            'Got {}'.format(self.which_set)
        )
        inputs = self.inputs
        targets = self.targets
        target_ids = self.target_ids

        kf = KFold(n_splits=k)
        # init list of DPs
        for train_index, val_index in kf.split(inputs):
            inputs_train, inputs_val = inputs[train_index], inputs[val_index]
            targets_train, targets_val = targets[train_index], targets[val_index]
            target_ids_train, targets_ids_val = target_ids[train_index], target_ids[val_index]

            train_data = {'inputs': inputs_train, 'targets': targets_train, 'target_ids': target_ids_train,
                          'max_num_ans': self.max_num_ans, 'max_prob_set_id': self.max_prob_set_id}
            val_data = {'inputs': inputs_val, 'targets': targets_val, 'target_ids': targets_ids_val,
                        'max_num_ans': self.max_num_ans, 'max_prob_set_id': self.max_prob_set_id}

            train_dp = ASSISTDataProvider(self.data_dir, self.which_set, self.which_year,
                                          self.batch_size, self.max_num_batches,
                                          self.shuffle_order, self.rng, data=train_data)
            val_dp = ASSISTDataProvider(self.data_dir, self.which_set, self.which_year,
                                        self.batch_size, self.max_num_batches,
                                        self.shuffle_order, self.rng, data=val_data)
            yield (train_dp, val_dp)

    def _validate_inputs(self, which_set, which_year, data_path):
        assert which_set in ['train', 'test'], (
            'Expected which_set to be either train or test. '
            'Got {0}'.format(which_set)
        )
        assert which_year in ['09', '15'], (
            'Expected which_year to be either 09 or 15. '
            'Got {}.format(which_year'
        )
        assert os.path.isfile(data_path + '-inputs.npz'), (
                'Data file does not exist at expected path: ' + data_path
        )
        assert os.path.isfile(data_path + '-targets.npz'), (
                'Data file does not exist at expected path: ' + data_path
        )
