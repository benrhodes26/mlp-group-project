from data_provider import ASSISTDataProvider
from LstmModel import LstmModel
from utils import get_events_filepath, events_to_numpy

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from time import gmtime, strftime

import os
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


START_TIME = strftime('%Y%m%d-%H%M', gmtime())

parser = ArgumentParser(description='Train LstmModel.',
                        formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_dir', type=str,
                    default='~/Dropbox/mlp-group-project/',
                    help='Path to directory containing data')
parser.add_argument('--which_set', type=str, default='train',
                    help='Either train or test')
parser.add_argument('--which_year', type=str, default='15',
                    help='Year of ASSIST data. Either 09 or 15')
parser.add_argument('--restore', default=None,
                    help='Path to .ckpt file of model to continue training')
parser.add_argument('--learn_rate',  type=float, default=0.01,
                    help='Initial learning rate for Adam optimiser')
parser.add_argument('--batch',  type=int, default=100,
                    help='Batch size')
parser.add_argument('--epochs', type=int, default=20,
                    help='Number of training epochs')
parser.add_argument('--decay', type=float, default=0.98,
                    help='Fraction to decay learning rate every 100 batches')
parser.add_argument('--keep_prob', type=float, default=1.0,
                    help='Fraction to keep in dropout applied to LSTM cell')
parser.add_argument('--use_plus_minus_feats', type=bool, default=False,
                    help='Whether or not to use +/-1s for feature encoding')
parser.add_argument('--compressed_sensing', type=bool, default=False,
                    help='Whether or not to use compressed sensing')
parser.add_argument('--fraction', type=float, default=1.0,
                    help='Fraction of data to use. Useful for hyerparam tuning')
parser.add_argument('--name', type=str, default=START_TIME,
                    help='Name of experiment when saving model')
parser.add_argument('--model_dir', type=str, default='.',
                    help='Path to directory where model will be saved')
args = parser.parse_args()

SAVE_DIR = os.path.join(args.model_dir, args.name)
os.mkdir(SAVE_DIR)

data_provider = ASSISTDataProvider(
    args.data_dir,
    which_set=args.which_set,
    which_year=args.which_year,
    batch_size=args.batch,
    use_plus_minus_feats=args.use_plus_minus_feats,
    use_compressed_sensing=args.compressed_sensing,
    fraction=args.fraction)
train_set, val_set = data_provider.train_validation_split()

Model = LstmModel(max_time_steps=train_set.max_num_ans,
                  feature_len=train_set.encoding_dim,
                  n_distinct_questions=train_set.max_prob_set_id)

print('Experiment started at', START_TIME)
print("Building model...")
Model.build_graph(n_hidden_units=200, learning_rate=args.learn_rate,
                  decay_exp=args.decay)
print("Model built!")

train_saver = tf.train.Saver()
valid_saver = tf.train.Saver()

with tf.Session() as sess:
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(SAVE_DIR+'/train', graph=sess.graph)
    valid_writer = tf.summary.FileWriter(SAVE_DIR+'/valid', graph=sess.graph)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())  # required for metrics

    if args.restore:
        train_saver.restore(sess, tf.train.latest_checkpoint(args.restore))
        print("Model restored!")

    print("Starting training...")
    for epoch in range(args.epochs):
        # Train one epoch!
        loss_total = 0
        accuracy_total = 0
        auc_total = 0
        for i, (inputs, targets, target_ids) in enumerate(train_set):
            _, loss, (accuracy, _), (auc, _), summary = sess.run(
                [Model.training, Model.loss, Model.accuracy, Model.auc, merged],
                feed_dict={Model.inputs: inputs,
                           Model.targets: targets,
                           Model.target_ids: target_ids,
                           Model.keep_prob: float(args.keep_prob)})

            loss_total += loss
            accuracy_total += accuracy
            auc_total += auc
        print("Epoch {},  Loss: {:.3f},  Accuracy: {:.3f},  AUC: {:.3f} (train)"
              .format(epoch, loss_total/(i+1), accuracy_total/(i+1),
                      auc_total/(i+1)))
        train_writer.add_summary(summary, epoch)

        # save model each epoch
        save_file = "{}/{}_{}.ckpt".format(SAVE_DIR, args.name, epoch)
        train_saver.save(sess, save_file)

        # Compute metrics on validation set (no training)
        loss_total = 0
        accuracy_total = 0
        auc_total = 0
        for i, (inputs, targets, target_ids) in enumerate(val_set):
            loss, (accuracy, _), (auc, _), summary = sess.run(
                [Model.loss, Model.accuracy, Model.auc, merged],
                feed_dict={Model.inputs: inputs,
                           Model.targets: targets,
                           Model.target_ids: target_ids,
                           Model.keep_prob: 1.0})
            loss_total += loss
            accuracy_total += accuracy
            auc_total += auc
        print("Epoch {},  Loss: {:.3f},  Accuracy: {:.3f},  AUC: {:.3f} (valid)"
              .format(epoch, loss_total/(i+1), accuracy_total/(i+1),
                      auc_total/(i+1)))
        valid_writer.add_summary(summary, epoch)
    print("Saved model at", save_file)  # training finished

    # Get and save loss, accuracy, and auc metrics
    events_file_train = get_events_filepath(SAVE_DIR, 'train')
    metrics_train = events_to_numpy(events_file_train)
    np.save(os.path.join(SAVE_DIR, 'metrics_train'), metrics_train)

    events_file_valid = get_events_filepath(SAVE_DIR, 'valid')
    metrics_valid = events_to_numpy(events_file_valid)
    np.save(os.path.join(SAVE_DIR, 'metrics_valid'), metrics_valid)

    # plot metrics
    e = np.arange(args.epochs)
    plt.figure()
    plt.plot(e, metrics_train[:, 0])
    plt.plot(e, metrics_valid[:, 0])
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.title('Loss per epoch')
    plt.savefig(SAVE_DIR + '/loss.png')

    plt.figure()
    plt.plot(e, metrics_train[:, 1])
    plt.plot(e, metrics_valid[:, 1])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per epoch')
    plt.savefig(SAVE_DIR + '/accuracy.png')

    plt.figure()
    plt.plot(e, metrics_train[:, 2])
    plt.plot(e, metrics_valid[:, 2])
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('AUC per epoch')
    plt.savefig(SAVE_DIR + '/auc.png')
