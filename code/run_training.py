from data_provider import ASSISTDataProvider
from LstmModel import LstmModel
from PlotResult import PlotResult

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from time import gmtime, strftime

import os
import numpy as np
import tensorflow as tf

#global graph
START_TIME = strftime('%Y%m%d-%H%M', gmtime())


def Arguments():
    parser = ArgumentParser(description='Train LstmModel.',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str,
                        default='~/Dropbox/mlp-group-project/',
                        help='Path to directory containing data')
    parser.add_argument('--which_set', type=str,
                        default='train', help='Either train or test')
    parser.add_argument(
        '--which_year', type=str, default='15',
        help='Year of ASSIST data. Either 09 or 15')
    parser.add_argument('--restore', default=None,
                        help='Path to .ckpt file of model to continue training')
    parser.add_argument('--learn_rate',  type=float, default=0.01,
                        help='Initial learning rate for Adam optimiser')
    parser.add_argument('--batch',  type=int, default=100,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument(
        '--decay', type=float, default=0.98,
        help='Fraction to decay learning rate every 100 batches')
    parser.add_argument('--use_plus_minus_feats', type=bool, default=False,
                        help='Whether or not to use +/-1s for feature encoding')
    parser.add_argument('--compressed_sensing', type=bool, default=False,
                        help='Whether or not to use compressed sensing')
    parser.add_argument('--name', type=str, default=START_TIME,
                        help='Name of experiment when saving model')
    parser.add_argument('--model_dir', type=str, default='.',
                        help='Path to directory where model will be saved')
    return parser.parse_args()


class DataSet(object):
    # Input Data class
    def __init__(
            self,
            data,
            max_time_steps,
            feature_len,
            n_distinct_questions,
            name=None):
        self.data = data
        self.max_time_steps = max_time_steps
        self.feature_len = feature_len
        self.n_distinct_questions = n_distinct_questions
        self.name = name


def GetDataSet(args):
    "Getting training and validation DataProviders."

    training_set_before_split = ASSISTDataProvider(
        args.data_dir, which_set=args.which_set, which_year=args.which_year,
        batch_size=args.batch, use_plus_minus_feats=args.use_plus_minus_feats,
        use_compressed_sensing=args.compressed_sensing)

    max_time_steps = training_set_before_split.max_num_ans
    feature_len = training_set_before_split.encoding_dim
    n_distinct_questions = training_set_before_split.max_prob_set_id

    for train, val in training_set_before_split.get_k_folds(5):
        train_set, val_set = train, val
        break

    train = DataSet(
        train_set,
        max_time_steps,
        feature_len,
        n_distinct_questions,
        name="Train")
    valid = DataSet(
        val_set,
        max_time_steps,
        feature_len,
        n_distinct_questions,
        name="Valid")

    return train, valid


def run_epochs(
        sess,
        model,
        dataset,
        summary_merge,
        data_inputs,
        data_targets,
        data_target_ids,
        model_op=None):

    fetches = [model.loss, model.accuracy, model.auc, summary_merge]

    # For training model, add optimizer
    if model_op is not None:
        fetches.append(model_op)

    for i, (inputs, targets, target_ids) in enumerate(dataset):

        state = sess.run(
                fetches,
                feed_dict={data_inputs: inputs,
                           data_targets: targets,
                           data_target_ids: target_ids})
    loss = state[0]
    (accuracy, _) = state[1]
    (auc, _) = state[2]
    summary = state[3]
    return loss, accuracy, auc, summary


def Plotting(event_dir, epochs, save_dir, train_result, valid_result):
    event_file = ""
    for (path, names, files) in os.walk(event_dir):
        event_file = event_dir+'/'+files[0]
        break
    PlotResult(event_file, epochs, save_dir, train_result, valid_result)


def main(_):

    args = Arguments()
    train_set, valid_set = GetDataSet(args)

    print('Experiment started at', START_TIME)
    print("Building model...")

    g = tf.Graph()
    with g.as_default():
        initializer = tf.random_uniform_initializer(-0.1,
                                                    0.1)
        data_inputs = tf.placeholder(
                tf.float32,
                shape=[None, train_set.max_time_steps, train_set.feature_len],
                name='inputs')
        data_targets = tf.placeholder(tf.float32,
                                      shape=[None],
                                      name='targets')
        data_target_ids = tf.placeholder(tf.int32,
                                         shape=[None],
                                         name='target_ids')
        with tf.name_scope("Train"):
            model_train = LstmModel(
                max_time_steps=train_set.max_time_steps,
                feature_len=train_set.feature_len,
                n_distinct_questions=train_set.n_distinct_questions,
                is_training=True)

            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                model_train.build_graph(
                    n_hidden_units=200,
                    learning_rate=args.learn_rate,
                    decay_exp=args.decay,
                    inputs=data_inputs,
                    targets=data_targets,
                    target_ids=data_target_ids)

            tf.summary.scalar("train_loss", model_train.loss)
            tf.summary.scalar("train_accuracy", model_train.accuracy[0])
            tf.summary.scalar("train_auc", model_train.auc[0])

        with tf.name_scope("Valid"):
            model_valid = LstmModel(
                max_time_steps=train_set.max_time_steps,
                feature_len=train_set.feature_len,
                n_distinct_questions=train_set.n_distinct_questions,
                is_training=False)

            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                model_valid.build_graph(
                    n_hidden_units=200,
                    inputs=data_inputs,
                    targets=data_targets,
                    target_ids=data_target_ids)
            tf.summary.scalar("valid_loss", model_valid.loss)
            tf.summary.scalar("valid_accuracy", model_valid.accuracy[0])
            tf.summary.scalar("valid_auc", model_valid.auc[0])

        train_saver = tf.train.Saver()

        tf.add_to_collection("optimizer", model_train.training)
        glob_init = tf.global_variables_initializer()
        local_init = tf.local_variables_initializer()

        merged = tf.summary.merge_all()

    save_dir = args.model_dir+'/'+args.name
    os.mkdir(save_dir)
    print("Model built!")

    train_result = []
    valid_result = []

    # Training over epochs
    with g.as_default():
        with tf.Session() as sess:
            train_writer = tf.summary.FileWriter(save_dir+'/train', sess.graph)
            sess.run(glob_init)
            sess.run(local_init)  # required for metrics

            if args.restore:
                train_saver.restore(
                    sess, tf.train.latest_checkpoint(
                        args.restore))
                print("Model restored!")

            print("Starting training...")
            for epoch in range(args.epochs):
                train_loss, train_accuracy, train_auc, train_summary = run_epochs(
                    sess, model_train, train_set.data, merged, data_inputs, data_targets, data_target_ids, model_op=model_train.training)

                valid_loss, valid_accuracy, valid_auc, valid_summary = run_epochs(
                    sess, model_train, valid_set.data, merged, data_inputs, data_targets, data_target_ids)
                train_writer.add_summary(train_summary)
                train_writer.add_summary(valid_summary)

                print(
                    "Epoch {}\nTrain:  Loss = {:.3f},  Accuracy = {:.3f},  AUC = {:.3f}\nValid:  Loss = {:.3f},  Accuracy = {:.3f},  AUC = {:.3f}" .format(
                        epoch,
                        train_loss,
                        train_accuracy,
                        train_auc,
                        valid_loss,
                        valid_accuracy,
                        valid_auc))
                train_result.append([train_loss, train_accuracy, train_auc])
                valid_result.append([valid_loss, valid_accuracy, valid_auc])

                save_path = "{}/{}_{}.ckpt".format(save_dir, args.name, epoch)
                train_saver.save(sess, save_path)

            train_result = np.array(train_result)
            valid_result = np.array(valid_result)

        print("Saved model at", save_path)

    # Save figure of loss, accuracy, auc graph
    event_dir = train_writer.get_logdir()
    train_writer.close()

    Plotting(event_dir, args.epochs, save_dir, train_result, valid_result)


if __name__ == "__main__":
    tf.app.run()
