from data_provider import ASSISTDataProvider
from LstmModel import LstmModel
from utils import get_events_filepath, events_to_numpy, get_learning_rate
from rnn import RNN,
from util import semiSortedMiniBatches

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from time import gmtime, strftime

import os
import numpy as np
import math
import tensorflow as tf
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt

START_TIME = strftime('%Y%m%d-%H%M', gmtime())

parser = ArgumentParser(description='Train LstmModel.',
                        formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_dir', type=str,
                    default='/afs/inf.ed.ac.uk/user/s17/s1771906/MLP/mlp-group-project/data',
                    help='Path to directory containing data')
parser.add_argument('--which_set', type=str, default='train',
                    help='Either train or test')
parser.add_argument('--which_year', type=str, default='09',
                    help='Year of ASSIST data. Either 09 or 15')
parser.add_argument('--restore', default=None,
                    help='Path to .ckpt file of model to continue training')
parser.add_argument('--optimisation', type=str, default='sgd',
                    help='optimisation method. Choices are: adam, rmsprop, '
                         'momentum and sgd.')
parser.add_argument('--init_learn_rate', type=float, default=30,
                    help='Initial learning rate.')
parser.add_argument('--min_learn_rate', type=float, default=1,
                    help='minimum possible learning rate.')
parser.add_argument('--lr_decay_step', type=float, default=12,
                    help='Decrease learning rate every x epochs')
parser.add_argument('--lr_exp_decay', type=float, default=(1 / 3),
                    help='fraction to multiply learning rate by each step')
parser.add_argument('--num_hidden_units', type=int, default=200,
                    help='Number of hidden units in the LSTM cell')
parser.add_argument('--batch', type=int, default=50,
                    help='Batch size')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of training epochs')
parser.add_argument('--decay', type=float, default=0.96,
                    help='Fraction to decay learning rate every 100 batches')
parser.add_argument('--decay_step', type=int, default=3000,
                    help='Apply learning rate decay every x batches')
# parser.add_argument('--add_gradient_noise', type=float, default=1e-3,
#                    help='add gaussian noise with stdev=1e-3 to gradients')
parser.add_argument('--clip_norm', type=float, default=1,
                    help='clip norms of gradients')
parser.add_argument('--keep_prob', type=float, default=0.6,
                    help='Fraction to keep in dropout applied to LSTM cell')
parser.add_argument('--var_dropout', dest='var_dropout', action='store_true',
                    help='use variational dropout')
parser.add_argument('--no-var_dropout', dest='var_dropout', action='store_false',
                    help='do not use variational dropout')
parser.set_defaults(var_dropout=True)
parser.add_argument('--plus_minus_feats', dest='plus_minus_feats', action='store_true',
                    help='use +/- for feature encoding')
parser.add_argument('--no-plus_minus_feats', dest='plus_minus_feats', action='store_false',
                    help='do not use +/- for feature encoding')
parser.set_defaults(plus_minus_feats=False)
parser.add_argument('--compressed_sensing', dest='compressed_sensing', action='store_true',
                    help='use compressed sensing')
parser.add_argument('--no-compressed_sensing', dest='compressed_sensing', action='store_false',
                    help='do not use use compressed sensing')
parser.set_defaults(compressed_sensing=False)
parser.add_argument('--log_stats', dest='log_stats', action='store_true',
                    help='print learning rate and gradient norms once an epoch')
parser.add_argument('--no-log_stats', dest='log_stats', action='store_false',
                    help='do not print learning rate and gradient norms once an epoch')
parser.set_defaults(log_stats=False)
parser.add_argument('--fraction', type=float, default=1.0,
                    help='Fraction of data to use. Useful for hyperparameter tuning')
parser.add_argument('--name', type=str, default=START_TIME,
                    help='Name of experiment when saving model')
parser.add_argument('--model_dir', type=str, default='.',
                    help='Path to directory where model will be saved')
args = parser.parse_args()

OUTPUT_DIR = os.path.join(args.model_dir, args.name)
os.mkdir(OUTPUT_DIR)

mini_batch_size = 100

data_provider = ASSISTDataProvider(
    args.data_dir,
    which_set=args.which_set,
    which_year=args.which_year,
    batch_size=args.mini_batch_size,
    use_plus_minus_feats=args.plus_minus_feats,
    use_compressed_sensing=args.compressed_sensing,
    fraction=args.fraction)
train_set, val_set = data_provider.train_validation_split()

START_EPOCH =  1
LEARNING_RATES = [30, 30, 30, 10, 10, 10, 5, 5, 5]
LEARNING_RATE_REPEATS = 4
MIN_LEARNING_RATE = 1

def run():
    n_hidden = 200
    decay_rate = 1.0
    init_rate = 30

    dropoutPred = True
    max_grad = 5e-5


    print('making rnn...')

    rnn = RNN(
                dropoutPred = dropoutPred,
                n_hidden = n_hidden,
                n_questions = data_provider.max_num_ans,
                maxGrad = max_grad,
                maxSteps = 4290,
                encoding_dim=train_set.encoding_dim,
                mini_batch_size = mini_batch_size)
    print("rnn made!")

    trainMiniBatch(rnn, train_set, val_set, args.batch)

def trainMiniBatch(rnn, train_set, val_set, mini_batch_size):
    print('train')

    epochIndex = START_EPOCH
    blob_size = 50
    
    while True:
        rate = getLearningRate(epochIndex)
        startTime = os.time()
        miniBatches = train_set
        totalTests = train_set.num_batches*train_set.batch_size
        sumErr = 0
        numTests = 0
        done = 0
        ####**** Setting Gradient Zero rnn:zeroGrad(350)
        miniTests = 0
        miniErr = 0

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            print("Starting training...")
            for epoch in range(args.epochs):
                # Train one epoch!
                sess.run(rnn.auc_init)
                sess.run(rnn.acc_init)

                for i, (inputs, targets, target_ids) in enumerate(train_set):
                    alpha = blob_size / totalTests
                    rnn.calcGrad()
                    _, loss, acc_update, auc_update, summary_loss, logit_list = sess.run(
                        [Model.training, Model.loss, Model.accuracy[1], Model.auc[1],
                         merged_loss, Model.logit_list],
                        feed_dict={Model.inputs: inputs,
                                   Model.targets: targets,
                                   Model.target_ids: target_ids,
                                   Model.learning_rate: learning_rate,
                                   Model.keep_prob: float(args.keep_prob)})


        for i, batch in enumerate(miniBatches)


            err, tests, maxNorm = rnn:calcGrad(batch, rate, alpha)
            sumErr = sumErr + err
            numTests = numTests + tests
            collectgarbage()
            done = done + blob_size
            miniErr = miniErr + err
            miniTests = miniTests + tests
            if done % mini_batch_size == 0 then
                rnn:update(350, rate)
                rnn:zeroGrad(350)
                print('trainMini', i /  # miniBatches, miniErr/miniTests, sumErr/numTests, rate)
                miniErr = 0
                miniTests = 0

def getLearningRate(epochIndex):
    rate = MIN_LEARNING_RATE
    rateIndex = math.floor((epochIndex - 1) / LEARNING_RATE_REPEATS) + 1
    if rateIndex <=  len(LEARNING_RATES):
            rate = LEARNING_RATES[rateIndex]

    return rate