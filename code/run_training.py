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


LEARNING_RATES = [30.0, 30.0, 30.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0]
MIN_LEARNING_RATE = 1

def GetLearningRate(epochIndex, LEARNING_RATE_REPEATS):
    rate = MIN_LEARNING_RATE
    rateIndex = math.floor((epochIndex - 1) / LEARNING_RATE_REPEATS) + 1
    if rateIndex <=  len(LEARNING_RATES):
            rate = LEARNING_RATES[int(rateIndex)]

    return rate


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
parser.add_argument('--batch', type=int, default=32,
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
train_set, val_set = data_provider.train_validation_split()


repeats = args.epochs/9 
total_train_num = train_set.max_num_ans*train_set.num_batches
total_valid_num = val_set.max_num_ans*val_set.num_batches

Model = LstmModel(max_time_steps=train_set.max_num_ans,
                  feature_len=train_set.encoding_dim,
                  n_distinct_questions=train_set.max_prob_set_id,
                  var_dropout=args.var_dropout)

print('Experiment started at', START_TIME)
print("Building model...")
Model.build_graph(n_hidden_units=args.num_hidden_units,
                  clip_norm=args.clip_norm,
                  # add_gradient_noise=args.add_gradient_noise,
                  optimisation=args.optimisation)
print("Model built!")

train_saver = tf.train.Saver()
valid_saver = tf.train.Saver()

with tf.Session() as sess:
    merged_loss = tf.summary.merge(Model.summary_loss)
    merged_aucacc = tf.summary.merge(Model.summary_aucacc)
    blob_size = 50
    train_writer = tf.summary.FileWriter(SAVE_DIR + '/train', graph=sess.graph)
    valid_writer = tf.summary.FileWriter(SAVE_DIR + '/valid', graph=sess.graph)
    sess.run(tf.global_variables_initializer())

    if args.restore:
        train_saver.restore(sess, tf.train.latest_checkpoint(args.restore))
        print("Model restored!")



    print("Starting training...")
    for epoch in range(args.epochs):
        # Train one epoch!
        sess.run(Model.auc_init)
        sess.run(Model.acc_init)
        total_sum = 0
        total_num = 0
        '''
        learning_rate = get_learning_rate(epoch,
                                          args.init_learn_rate,
                                          args.min_learn_rate,
                                          args.lr_exp_decay,
                                          args.lr_decay_step)
        '''

        learning_rate = GetLearningRate(float(epoch), repeats)
        print('Learning Rate : ', learning_rate)
        for i, (inputs, targets, target_ids) in enumerate(train_set):
            alpha = blob_size / total_train_num

            _, loss, acc_update, auc_update, summary_loss,logit_list = sess.run(
                [Model.training, Model.loss, Model.accuracy[1], Model.auc[1],
                 merged_loss,Model.logit_list],
                feed_dict={Model.inputs: inputs,
                           Model.targets: targets,
                           Model.target_ids: target_ids,
                           Model.learning_rate: learning_rate,
                           Model.alpha: alpha,
                           Model.keep_prob: float(args.keep_prob)})



            if args.log_stats and i == 0:
                # optional logging for debugging.
                print("learning rate is: {}".format(learning_rate))
                for gv in Model.grads_and_vars:
                    _ = sess.run([tf.Print(gv, [tf.norm(gv[0]), gv[1].name],
                                           message="Grad norm is: ")],
                                 feed_dict={Model.inputs: inputs,
                                            Model.targets: targets,
                                            Model.target_ids: target_ids,
                                            Model.learning_rate: learning_rate,
                                            Model.keep_prob: float(args.keep_prob)})

        accuracy, auc, summary_aucacc = sess.run(
            [Model.accuracy[0], Model.auc[0], merged_aucacc],
            feed_dict={Model.inputs: inputs,
                       Model.targets: targets,
                       Model.target_ids: target_ids})

        predict = logit_list[0]['prediction']
        logits = logit_list[0]['logits']
        target = logit_list[0]['target']
        compare = np.array([target, predict, logits]).T
        total_sum += np.sum(target - predict != 0)
        total_num += len(target)
        print(
            "Epoch {},  Loss: {:.3f},  Total_sum:{:.3f}, Accuracy: {:.3f},  AUC: {:.3f} (train)"
                .format(epoch, loss, total_sum / total_num, accuracy, auc))


        train_writer.add_summary(summary_loss, epoch)
        train_writer.add_summary(summary_aucacc, epoch)

        # save model each epoch
        save_file = "{}/{}_{}.ckpt".format(SAVE_DIR, args.name, epoch)
        train_saver.save(sess, save_file)

        sess.run(Model.auc_init)
        sess.run(Model.acc_init)

        # Compute metrics on validation set (no training)
        loss_total = 0
        accuracy_total = 0
        auc_total = 0

        for i, (inputs, targets, target_ids) in enumerate(val_set):
            alpha = blob_size / total_valid_num

            loss, acc_update, auc_update, summary_loss = sess.run(
                [Model.loss, Model.accuracy[1], Model.auc[1], merged_loss],
                feed_dict={
                    Model.inputs: inputs,
                    Model.targets: targets,
                    Model.target_ids: target_ids})

            accuracy, auc, summary_aucacc = sess.run(
                [Model.accuracy[0], Model.auc[0], merged_aucacc],
                feed_dict={Model.inputs: inputs,
                           Model.targets: targets,
                           Model.target_ids: target_ids,
                           Model.alpha: alpha,
                           Model.keep_prob: 1.0})
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

