from data_provider import ASSISTDataProvider
from LstmModel import LstmModel
from utils import get_events_filepath, events_to_numpy, get_learning_rate

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from time import gmtime, strftime

import os
import numpy as np
import tensorflow as tf
import matplotlib
import math

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
parser.add_argument('--init_learn_rate', type=float, default=10,
                    help='Initial learning rate.')
parser.add_argument('--min_learn_rate', type=float, default=1,
                    help='minimum possible learning rate.')
parser.add_argument('--lr_decay_step', type=float, default=12,
                    help='Decrease learning rate every x epochs')
parser.add_argument('--lr_exp_decay', type=float, default=0.5,
                    help='fraction to multiply learning rate by each step')
parser.add_argument('--epsilon', type=float, default=0.1,
                    help='epsilon value for adam optimisers')
parser.add_argument('--num_hidden_units', type=int, default=200,
                    help='Number of hidden units in the LSTM cell')
parser.add_argument('--batch', type=int, default=96,
                    help='Batch size')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of training epochs')
parser.add_argument('--threshold', type=int, default=None,
                    help='threshold sequence lengths')
# parser.add_argument('--decay', type=float, default=0.96,
#                   help='Fraction to decay learning rate every 100 batches')
# parser.add_argument('--decay_step', type=int, default=3000,
#                   help='Apply learning rate decay every x batches')
# parser.add_argument('--add_gradient_noise', type=float, default=1e-3,
#                    help='add gaussian noise with stdev=1e-3 to gradients')
parser.add_argument('--clip_norm', type=float, default=1,
                    help='clip norms of gradients')
parser.add_argument('--keep_prob', type=float, default=0.6,
                    help='Fraction to keep in dropout applied to LSTM cell')
parser.add_argument('--var_dropout', dest='var_dropout', action='store_true',
                    help='use variational dropout')
parser.add_argument('--plus_minus_feats', dest='plus_minus_feats', action='store_true',
                    help='use +/- for feature encoding')
parser.add_argument('--compressed_sensing', dest='compressed_sensing', action='store_true',
                    help='use compressed sensing')
parser.add_argument('--log_stats', dest='log_stats', action='store_true',
                    help='print learning rate and gradient norms once an epoch')
parser.add_argument('--fraction', type=float, default=1.0,
                    help='Fraction of data to use. Useful for hyperparameter tuning')
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
    use_plus_minus_feats=args.plus_minus_feats,
    use_compressed_sensing=args.compressed_sensing,
    fraction=args.fraction)
train_set, val_set = data_provider.train_validation_split(threshold=args.threshold)


repeats = args.epochs/9
total_train_num = train_set.max_num_ans*train_set.num_batches
total_valid_num = val_set.max_num_ans*val_set.num_batches
'''
Model = LstmModel(max_time_steps=train_set.max_num_ans,
                  feature_len=train_set.encoding_dim,
                  n_distinct_questions=train_set.max_prob_set_id,
                  var_dropout=args.var_dropout)
'''

print('Experiment started at', START_TIME)

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True,
                                  log_device_placement=False)
    with tf.Session(config=session_conf) as sess:

        initializer = tf.random_uniform_initializer(-0.05, 0.05)
        print("var_dropout: ", args.var_dropout)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            TrainModel = LstmModel(max_time_steps=train_set.max_num_ans,
                              feature_len=train_set.encoding_dim,
                              n_distinct_questions=train_set.max_prob_set_id,
                              var_dropout=args.var_dropout)
            TrainModel.build_graph(n_hidden_units=args.num_hidden_units,
                              clip_norm=args.clip_norm,
                              # add_gradient_noise=args.add_gradient_noise,
                              optimisation=args.optimisation,
                              is_training = True)
            TrainModel._build_metrics()


        with tf.variable_scope("model", reuse=True, initializer=initializer):
            TestModel = LstmModel(max_time_steps=train_set.max_num_ans,
                              feature_len=train_set.encoding_dim,
                              n_distinct_questions=train_set.max_prob_set_id,
                              var_dropout=args.var_dropout)
            TestModel.build_graph(n_hidden_units=args.num_hidden_units,
                              clip_norm=args.clip_norm,
                              # add_gradient_noise=args.add_gradient_noise,
                              optimisation=args.optimisation,
                              is_training = False)
            TestModel._build_metrics()

        train_saver = tf.train.Saver()
        valid_saver = tf.train.Saver()

        train_merged_loss = tf.summary.merge(TrainModel.summary_loss)
        train_merged_aucacc = tf.summary.merge(TrainModel.summary_aucacc)

        test_merged_loss = tf.summary.merge(TestModel.summary_loss)
        test_merged_aucacc = tf.summary.merge(TestModel.summary_aucacc)

        train_writer = tf.summary.FileWriter(SAVE_DIR + '/train', graph=sess.graph)
        valid_writer = tf.summary.FileWriter(SAVE_DIR + '/valid', graph=sess.graph)

        #####sess.run(tf.initialize_all_variables())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        if args.restore:
            train_saver.restore(sess, tf.train.latest_checkpoint(args.restore))
            print("Model restored!")


        print("Starting training...")
        for epoch in range(args.epochs):
            # Train one epoch!
            sess.run(TrainModel.auc_init)
            sess.run(TrainModel.acc_init)
            total_sum = 0
            total_num = 0

            learning_rate = get_learning_rate(epoch,
                                              args.init_learn_rate,
                                              args.min_learn_rate,
                                              args.lr_exp_decay,
                                              args.lr_decay_step)

            print('Learning Rate : ', learning_rate)
            for i, (inputs, targets, target_ids, seq_len) in enumerate(train_set):
                
                _, loss, acc_update, auc_update, summary_loss,logit_list = sess.run(
                    [TrainModel.training, TrainModel.loss, TrainModel.accuracy[1], TrainModel.auc[1],
                     train_merged_loss,TrainModel.logit_list],
                    feed_dict={TrainModel.inputs: inputs,
                               TrainModel.targets: targets,
                               TrainModel.target_ids: target_ids,
                               TrainModel.learning_rate: learning_rate,
                               TrainModel.keep_prob: float(args.keep_prob),
                               TrainModel.seq_len: seq_len})


                if args.log_stats and i == 0:
                    # optional logging for debugging.
                    print("learning rate is: {}".format(learning_rate))
                    for gv in TrainModel.grads_and_vars:
                        _ = sess.run([tf.Print(gv, [tf.norm(gv[0]), gv[1].name],
                                               message="Grad norm is: ")],
                                     feed_dict={TrainModel.inputs: inputs,
                                                TrainModel.targets: targets,
                                                TrainModel.target_ids: target_ids,
                                                TrainModel.learning_rate: learning_rate,
                                                TrainModel.keep_prob: float(args.keep_prob)})

                accuracy, auc, summary_aucacc = sess.run(
                    [TrainModel.accuracy[0], TrainModel.auc[0], train_merged_aucacc],
                    feed_dict={TrainModel.inputs: inputs,
                               TrainModel.targets: targets,
                               TrainModel.target_ids: target_ids})


            print(
                "Epoch {},  Loss: {:.3f}, Accuracy: {:.3f},  AUC: {:.3f} (train)"
                    .format(epoch, loss,accuracy, auc))


            train_writer.add_summary(summary_loss, epoch)
            train_writer.add_summary(summary_aucacc, epoch)

            # save model each epoch
            save_file = "{}/{}_{}.ckpt".format(SAVE_DIR, args.name, epoch)
            train_saver.save(sess, save_file)

            sess.run(TestModel.auc_init)
            sess.run(TestModel.acc_init)

            # Compute metrics on validation set (no training)
            loss_total = 0
            accuracy_total = 0
            auc_total = 0

            for i, (inputs, targets, target_ids, seq_len) in enumerate(val_set):

                loss, acc_update, auc_update, summary_loss = sess.run(
                    [TestModel.loss, TestModel.accuracy[1], TestModel.auc[1], test_merged_loss],
                    feed_dict={
                        TestModel.inputs: inputs,
                        TestModel.targets: targets,
                        TestModel.target_ids: target_ids,
                        TestModel.seq_len: seq_len})

                accuracy, auc, summary_aucacc = sess.run(
                    [TestModel.accuracy[0], TestModel.auc[0], test_merged_aucacc],
                    feed_dict={TestModel.inputs: inputs,
                               TestModel.targets: targets,
                               TestModel.target_ids: target_ids,
                               TestModel.keep_prob: 1.0})
            print("Epoch {},  Loss: {:.3f},  Accuracy: {:.3f},  AUC: {:.3f} (valid)"
                    .format(epoch,loss, accuracy, auc))

            valid_writer.add_summary(summary_loss, epoch)
            valid_writer.add_summary(summary_aucacc, epoch)

        train_writer.close()
        valid_writer.close()

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
        train_plt, = plt.plot(e, metrics_train[:, 0])
        valid_plt, = plt.plot(e, metrics_valid[:, 0])
        plt.legend([train_plt, valid_plt], ['train', 'valid'])
        plt.xlabel('Epoch')
        plt.ylabel('loss')
        plt.title('Loss per epoch')
        plt.savefig(SAVE_DIR + '/loss.png')

        plt.figure()
        train_plt, = plt.plot(e, metrics_train[:, 1])
        valid_plt, = plt.plot(e, metrics_valid[:, 1])
        plt.legend([train_plt, valid_plt], ['train', 'valid'])
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.title('AUC per epoch')
        plt.savefig(SAVE_DIR + '/auc.png')

        plt.figure()
        train_plt, = plt.plot(e, metrics_train[:, 2])
        valid_plt, = plt.plot(e, metrics_valid[:, 2])
        plt.legend([train_plt, valid_plt], ['train', 'valid'])
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy per epoch')
        plt.savefig(SAVE_DIR + '/accuracy.png')
