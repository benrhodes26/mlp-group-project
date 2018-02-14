from data_provider import ASSISTDataProvider
from LstmModel import LstmModel

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
parser.add_argument('--which_set', type=str,
                    default='train', help='Either train or test')
parser.add_argument('--which_year', type=str,
                    default='15', help='Year of ASSIST data. Either 09 or 15')
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
parser.add_argument('--use_plus_minus_feats', type=bool, default=False,
                    help='Whether or not to use +/-1s for feature encoding')
parser.add_argument('--compressed_sensing', type=bool, default=False,
                    help='Whether or not to use compressed sensing')
parser.add_argument('--name', type=str, default=START_TIME,
                    help='Name of experiment when saving model')
parser.add_argument('--model_dir', type=str, default='.',
                    help='Path to directory where model will be saved')
args = parser.parse_args()

training_set_before_split = ASSISTDataProvider(
    args.data_dir,
    which_set=args.which_set,
    which_year=args.which_year,
    batch_size=args.batch,
    use_plus_minus_feats=args.use_plus_minus_feats,
    use_compressed_sensing=args.compressed_sensing)

max_time_steps = training_set_before_split.max_num_ans
feature_len = training_set_before_split.encoding_dim
n_distinct_questions = training_set_before_split.max_prob_set_id

for train, val in training_set_before_split.get_k_folds(5):
    train_set, val_set = train, val
    break

Model = LstmModel(max_time_steps=max_time_steps, feature_len=feature_len,
                  n_distinct_questions=n_distinct_questions)

print('Experiment started at', START_TIME)
print("Building model...")
Model.build_graph(n_hidden_units=200, learning_rate=args.learn_rate,
                  decay_exp=args.decay)
print("Model built!")

save_dir = os.path.join(args.model_dir, args.name)
os.mkdir(save_dir)
train_saver = tf.train.Saver()
valid_saver = tf.train.Saver()

with tf.Session() as sess:
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(save_dir+'/train', graph=sess.graph)
    valid_writer = tf.summary.FileWriter(save_dir+'/valid', graph=sess.graph)

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
                           Model.target_ids: target_ids})

            loss_total += loss
            accuracy_total += accuracy
            auc_total += auc
        print("Epoch {},  Loss: {:.3f},  Accuracy: {:.3f},  AUC: {:.3f} (train)"
              .format(epoch, loss_total/(i+1), accuracy_total/(i+1),
                      auc_total/(i+1)))
        train_writer.add_summary(summary, epoch)

        # save model each epoch
        save_file = "{}/{}_{}.ckpt".format(save_dir, args.name, epoch)
        train_saver.save(sess, save_file)

        # Compute metrics on validation set (no training)
        loss_total = 0
        accuracy_total = 0
        auc_total = 0
        for i, (inputs, targets, target_ids) in enumerate(val_set):
            loss, (accuracy, _), (auc, _) = sess.run(
                [Model.loss, Model.accuracy, Model.auc],
                feed_dict={Model.inputs: inputs,
                           Model.targets: targets,
                           Model.target_ids: target_ids})
            loss_total += loss
            accuracy_total += accuracy
            auc_total += auc
        print("Epoch {},  Loss: {:.3f},  Accuracy: {:.3f},  AUC: {:.3f} (valid)"
              .format(epoch, loss_total/(i+1), accuracy_total/(i+1),
                      auc_total/(i+1)))
        valid_writer.add_summary(summary, epoch)
    print("Saved model at", save_file)

    # Save figure of loss, accuracy, auc graph
    result = []
    for dataset in ('train', 'valid'):
        path = os.path.join(save_dir, dataset)
        event_filename = os.listdir(path)[0]
        event_file = os.path.join(path, event_filename)
        for event in tf.train.summary_iterator(event_file):
            value_set = []
            is_result = False
            for v in event.summary.value:
                if v.tag == 'loss' or v.tag == 'accuracy_1' or v.tag == 'auc_1':
                    value_set.append(v.simple_value)
                    is_result = True
            if is_result:
                result.append(value_set)

    result = np.array(result)
    np.save(save_dir + '/results-' + START_TIME, result)

    e = np.arange(1, args.epochs+1)
    plt.figure()
    plt.plot(e, result[:, 0])
    plt.plot(e, result[:, 3])
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.title('Loss per epoch')
    plt.savefig(save_dir + '/loss.png')

    plt.figure()
    plt.plot(e, result[:, 1])
    plt.plot(e, result[:, 4])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per epoch')
    plt.savefig(save_dir + '/accuracy.png')

    plt.figure()
    plt.plot(e, result[:, 2])
    plt.plot(e, result[:, 5])
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('AUC per epoch')
    plt.savefig(save_dir + '/auc.png')
