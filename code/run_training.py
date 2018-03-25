from data_provider import ASSISTDataProvider
from LstmModel import LstmModel
from utils import get_learning_rate, log_learning_rate_and_grad_norms, plot_learning_curves

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from time import gmtime, strftime

import os
import tensorflow as tf

START_TIME = strftime('%Y%m%d-%H%M', gmtime())

# Arguments for reading and writing data
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
parser.add_argument('--name', type=str, default=START_TIME,
                    help='Name of experiment when saving model')
parser.add_argument('--model_dir', type=str, default='.',
                    help='Path to directory where model will be saved')

# Arguments for debugging
parser.add_argument('--log_stats', dest='log_stats', action='store_true',
                    help='print learning rate and gradient norms once every 10 epochs')
parser.add_argument('--no-log_stats', dest='log_stats', action='store_false',
                    help='do not print learning rate and gradient norms once every 10 epochs')
parser.set_defaults(log_stats=False)
parser.add_argument('--fraction', type=float, default=1.0,
                    help='Fraction of data to use. Useful for hyperparameter tuning')

# Arguments controlling feature representation
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
parser.add_argument('--max_time_steps', type=int, default=None,
                    help='limit length of students sequences of answers')

# Arguments to control model hyperparameters
parser.add_argument('--optimisation', type=str, default='sgd',
                    help='optimisation method. Choices are: adam, rmsprop, '
                         'momentum and sgd.')
parser.add_argument('--init_learn_rate', type=float, default=10,
                    help='Initial learning rate.')
parser.add_argument('--min_learn_rate', type=float, default=1,
                    help='minimum possible learning rate.')
parser.add_argument('--lr_decay_step', type=float, default=10,
                    help='Decrease learning rate every x epochs')
parser.add_argument('--lr_exp_decay', type=float, default=0.5,
                    help='fraction to multiply learning rate by each step')
parser.add_argument('--num_hidden_units', type=int, default=200,
                    help='Number of hidden units in the LSTM cell')
parser.add_argument('--batch', type=int, default=32,
                    help='Batch size')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of training epochs')
parser.add_argument('--clip_norm', type=float, default=1,
                    help='clip norms of gradients')
parser.add_argument('--keep_prob', type=float, default=0.6,
                    help='Fraction to keep in dropout applied to LSTM cell')
parser.add_argument('--var_dropout', dest='var_dropout', action='store_true',
                    help='use variational dropout')
parser.add_argument('--no-var_dropout', dest='var_dropout', action='store_false',
                    help='do not use variational dropout')
parser.set_defaults(var_dropout=True)
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
train_set, val_set = data_provider.train_validation_split(args.max_time_steps)

model = LstmModel(max_time_steps=train_set.max_num_ans,
                  feature_len=train_set.encoding_dim,
                  n_distinct_questions=train_set.max_prob_set_id,
                  var_dropout=args.var_dropout,
                  batch_size=args.batch)

print('Experiment started at', START_TIME)

model.build_graph(n_hidden_units=args.num_hidden_units,
                  clip_norm=args.clip_norm,
                  optimisation=args.optimisation)

train_saver = tf.train.Saver()
valid_saver = tf.train.Saver()

with tf.Session() as sess:
    merged_loss = tf.summary.merge(model.summary_loss)
    merged_aucacc = tf.summary.merge(model.summary_aucacc)
    train_writer = tf.summary.FileWriter(SAVE_DIR + '/train', graph=sess.graph)
    valid_writer = tf.summary.FileWriter(SAVE_DIR + '/valid', graph=sess.graph)
    sess.run(tf.global_variables_initializer())

    if args.restore:
        train_saver.restore(sess, tf.train.latest_checkpoint(args.restore))
        print("Model restored!")

    print("Starting training...")
    for epoch in range(args.epochs):
        model.reuse = False
        sess.run(model.auc_init)
        sess.run(model.acc_init)
        learning_rate = get_learning_rate(epoch, args.init_learn_rate, args.min_learn_rate,
                                          args.lr_exp_decay, args.lr_decay_step)

        for i, (inputs, targets, target_ids) in enumerate(train_set):
            _, loss, acc_update, auc_update, summary_loss = sess.run(
                [model.training, model.loss, model.accuracy[1], model.auc[1],
                 merged_loss],
                feed_dict={model.inputs: inputs,
                           model.targets: targets,
                           model.target_ids: target_ids,
                           model.learning_rate: learning_rate,
                           model.keep_prob: float(args.keep_prob)})

            if args.log_stats and epoch % 10 == 0 and i == 0:
                log_learning_rate_and_grad_norms(sess, model, inputs, targets,
                                                 target_ids, learning_rate, args.keep_prob)

        accuracy, auc, summary_aucacc = sess.run(
            [model.accuracy[0], model.auc[0], merged_aucacc])

        print("Epoch {},  Loss: {:.3f},  Accuracy: {:.3f},  AUC: {:.3f} (train)"
              .format(epoch, loss, accuracy, auc))

        # save metrics and model each epoch
        train_writer.add_summary(summary_loss, epoch)
        train_writer.add_summary(summary_aucacc, epoch)
        save_file = "{}/{}_{}.ckpt".format(SAVE_DIR, args.name, epoch)
        train_saver.save(sess, save_file)

        # Evaluate on validation set
        model.reuse = True
        sess.run(model.auc_init)
        sess.run(model.acc_init)

        for i, (inputs, targets, target_ids) in enumerate(val_set):
            loss, acc_update, auc_update, summary_loss = sess.run(
                [model.loss, model.accuracy[1], model.auc[1], merged_loss],
                feed_dict={
                    model.inputs: inputs,
                    model.targets: targets,
                    model.target_ids: target_ids})

        accuracy, auc, summary_aucacc = sess.run(
            [model.accuracy[0], model.auc[0], merged_aucacc])
        print("Epoch {},  Loss: {:.3f},  Accuracy: {:.3f},  AUC: {:.3f} (valid)"
              .format(epoch, loss, accuracy, auc))

        valid_writer.add_summary(summary_loss, epoch)
        valid_writer.add_summary(summary_aucacc, epoch)

    train_writer.close()
    valid_writer.close()

    print("Saved model at", save_file)  # training finished

plot_learning_curves(SAVE_DIR, args.epochs)
