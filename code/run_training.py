from data_provider import ASSISTDataProvider
from LstmModel import LstmModel

import numpy as np
import tensorflow as tf


DATA_DIR = '~/Dropbox/mlp-group-project/'
BATCH_SIZE = 100
EPOCHS = 50

experiment_name = 'first'

Model = LstmModel()
TrainingSet = ASSISTDataProvider(DATA_DIR, batch_size=BATCH_SIZE)

print("Building model...")
Model.build_graph(n_hidden_units=200, learning_rate=0.1)
print("Model built!")

print("Starting training...")
with tf.Session() as sess:
    train_saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    losses = []

    for epoch in range(EPOCHS):
        for inputs, targets, target_ids in TrainingSet:
            # ensure shapes and types as model expects
            inputs = np.squeeze(np.array(inputs, dtype=np.float32))
            inputs = np.transpose(inputs, [1, 0, 2])
            targets = np.array(targets, dtype=np.float32)
            target_ids = np.array(target_ids, dtype=np.int32)

            # Train!
            _, loss = sess.run(
                [Model.training, Model.loss],
                feed_dict={Model.inputs: inputs,
                           Model.targets: targets,
                           Model.target_ids: target_ids})
            print("Training underway...")

        print("Epoch: {}, loss: {}".format(epoch, loss))

        # save model each epoch
        save_path = "./{}_{}.ckpt".format(
            experiment_name, epoch)
        train_saver.save(sess, save_path)
    print("Saved model at", save_path)
