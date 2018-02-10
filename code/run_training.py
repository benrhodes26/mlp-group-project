from data_provider import ASSISTDataProvider
from LstmModel import LstmModel

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from time import gmtime, strftime

import numpy as np
import tensorflow as tf


START_TIME = strftime('%Y%m%d-%H%M', gmtime())

parser = ArgumentParser(description='Train LstmModel.',
                        formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_dir', type=str,
                    default='~/Dropbox/mlp-group-project/',
                    help='Path to directory containing data')
parser.add_argument('--learn_rate',  type=float, default=0.01,
                    help='Initial learning rate for Adam optimiser')
parser.add_argument('--batch',  type=int, default=100,
                    help='Batch size')
parser.add_argument('--epochs', type=int, default=20,
                    help='Number of training epochs')
parser.add_argument('--decay', default=0.9,
                    help='Fraction to decay learning rate every 100 batches')
parser.add_argument('--name', type=str, default=START_TIME,
                    help='Name of experiment when saving model')
parser.add_argument('--model_dir', type=str, default='.',
                    help='Path to directory where model will be saved')
args = parser.parse_args()

Model = LstmModel()
TrainingSet = ASSISTDataProvider(args.data_dir, batch_size=args.batch)

print('Experiment started at', START_TIME)
print("Building model...")
Model.build_graph(n_hidden_units=200, learning_rate=args.learn_rate,
                  decay_exp=args.decay)
print("Model built!")

print("Starting training...")
with tf.Session() as sess:
    train_saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())  # required for metrics
    losses = []

    for epoch in range(args.epochs):
        for i, (inputs, targets, target_ids) in enumerate(TrainingSet):
            # ensure shapes and types as model expects
            inputs = np.squeeze(np.array(inputs, dtype=np.float32))
            inputs = np.transpose(inputs, [1, 0, 2])
            targets = np.array(targets, dtype=np.float32)
            target_ids = np.array(target_ids, dtype=np.int32)

            # Train!
            _, loss, acc, auc = sess.run(
                [Model.training, Model.loss, Model.accuracy, Model.auc],
                feed_dict={Model.inputs: inputs,
                           Model.targets: targets,
                           Model.target_ids: target_ids})
            print("Training underway... Batch: {}, loss: {}".format(i, loss))
            print("Accuracy: {}, AUC: {}".format(acc, auc))

        print("Epoch: {}, loss: {}".format(epoch, loss))

        # save model each epoch
        save_path = "{}/{}_{}.ckpt".format(
            args.model_dir, args.name, epoch)
        train_saver.save(sess, save_path)
    print("Saved model at", save_path)
