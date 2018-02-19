import tensorflow as tf


class LstmModel:

    def __repr__(self):
        return "LstmModel"

    def __init__(self, max_time_steps=973, feature_len=293,
                 n_distinct_questions=146):
        """Initialise task-specific parameters."""
        self.max_time_steps = max_time_steps
        self.feature_len = feature_len
        self.n_distinct_questions = n_distinct_questions
        self.acc_init = None
        self.auc_init = None
        self.summary_loss = None
        self.summary_aucacc = None

    def build_graph(
            self,
            n_hidden_layers=1,
            n_hidden_units=200,
            keep_prob=1.0,
            learning_rate=0.01,
            clip_norm=10.0,
            decay_exp=None,
            add_gradient_noise=1e-3):
        self._build_model(n_hidden_layers=n_hidden_layers,
                          n_hidden_units=n_hidden_units)
        self._build_training(learning_rate=learning_rate,
                             decay_exp=decay_exp,
                             clip_norm=clip_norm,
                             add_gradient_noise=add_gradient_noise)
        self._build_metrics()

    def _build_model(self, n_hidden_layers=1, n_hidden_units=200):
        """Build a TensorFlow computational graph for an LSTM network.

        Model based on "DKT paper" (see section 3):
            Piech, Chris, et al. "Deep knowledge tracing."
            Advances in Neural Information Processing Systems. 2015.

        Implementation based on "GD paper" (see section 3):
            Xiong, Xiaolu, et al. "Going Deeper with Deep Knowledge Tracing."
            EDM. 2016.


        Parameters
        ----------
        n_hidden_layers : int (default=1)
            A single hidden layer was used in DKT paper
        n_hidden_units : int (default=200)
            200 hidden units were used in DKT paper
        keep_prob : float in [0, 1] (default=1.0)
            Probability a unit is kept in dropout layer
        """
        tf.reset_default_graph()

        # data. 'None' means any length batch size accepted
        self.inputs = tf.placeholder(
            tf.float32,
            shape=[None, self.max_time_steps, self.feature_len],
            name='inputs')

        # 'None' because may have answered any number of questions
        self.targets = tf.placeholder(tf.float32,
                                      shape=[None],
                                      name='targets')

        # int type required for tf.gather function
        self.target_ids = tf.placeholder(tf.int32,
                                         shape=[None],
                                         name='target_ids')

        self.keep_prob = tf.placeholder_with_default(1.0, shape=(),
                                                     name='keep_prob')

        # model. LSTM layer(s) then linear layer (softmax applied in loss)
        cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                             output_keep_prob=self.keep_prob,
                                             state_keep_prob=self.keep_prob,
                                             variational_recurrent=True,
                                             dtype=tf.float32)
        if n_hidden_layers > 1:
            cells = [cell for layer in n_hidden_layers]
            cell = tf.nn.rnn_cell.MultiRNNCell(cells)

        with tf.variable_scope('RNN', initializer=tf.contrib.layers.xavier_initializer()):
            self.outputs, self.state = tf.nn.dynamic_rnn(cell=cell,
                                                         inputs=self.inputs,
                                                         dtype=tf.float32)

        sigmoid_w = tf.get_variable(dtype=tf.float32,
                                    name="sigmoid_w",
                                    shape=[n_hidden_units,
                                           self.n_distinct_questions])
        sigmoid_b = tf.get_variable(dtype=tf.float32,
                                    name="sigmoid_b",
                                    shape=[self.n_distinct_questions])

        # reshaping as done in GD paper code
        # first dim now batch_size times max_time_steps
        self.outputs = tf.reshape(self.outputs,
                                  shape=[-1, n_hidden_units])

        logits = tf.matmul(self.outputs, sigmoid_w) + sigmoid_b
        logits = tf.reshape(logits, [-1])
        self.logits = tf.dynamic_partition(logits, self.target_ids, 2)[1]
        self.predictions = tf.round(tf.nn.sigmoid(self.logits))

    def _build_training(self, learning_rate=0.001, decay_exp=None,
                        clip_norm=10.0, add_gradient_noise=1e-3):
        """Define parameters updates.

        Applies exponential learning rate decay (optional). See:
        https://www.tensorflow.org/versions/r0.12/api_docs/python/train
        /decaying_the_learning_rate

        Applies gradient clipping by global norm (optional). See:
        https://www.tensorflow.org/versions/r0.12/api_docs/python/train
        /gradient_clipping
        """
        loss_per_example = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.logits, labels=self.targets)
        self.loss = tf.reduce_mean(loss_per_example)
        self.summary_loss = [tf.summary.scalar('loss', self.loss)]
        # track number of batches seen
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        if decay_exp:  # decay every 3000 batches, roughly 2 epochs on 2015 data
            learning_rate = tf.train.exponential_decay(
                learning_rate=learning_rate, global_step=self.global_step,
                decay_rate=decay_exp, decay_steps=3000, staircase=True)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Ensures that we execute the update_ops before performing the
            # train_step
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            grads, trainable_vars = zip(*optimizer.compute_gradients(self.loss))

            if clip_norm:
                # grads, _ = tf.clip_by_global_norm(grads, clip_norm)
                grads = [tf.clip_by_norm(grad, clip_norm) for grad in grads]
            if add_gradient_noise:
                grads = [self.add_noise(g) for g in grads]

            self.training = optimizer.apply_gradients(
                zip(grads, trainable_vars),
                global_step=self.global_step)

    def _build_metrics(self):
        """Add ability to compute accuracy and AUC."""
        self.accuracy = tf.metrics.accuracy(labels=self.targets,
                                            predictions=self.predictions,
                                            name="acc")

        self.auc = tf.metrics.auc(labels=self.targets,
                                  predictions=self.predictions,
                                  name="auc")

        self.summary_aucacc = [
            tf.summary.scalar(
                'auc', self.auc[0]), tf.summary.scalar(
                'accuracy', self.accuracy[0])]

        auc_var = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="auc")
        self.auc_init = tf.variables_initializer(var_list=auc_var)

        acc_var = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="acc")
        self.acc_init = tf.variables_initializer(var_list=acc_var)

    def add_noise(self, t, stddev=1e-3, name=None):
        """
        This code taken directly from:
            Xiong, Xiaolu, et al. "Going Deeper with Deep Knowledge Tracing."
            EDM. 2016.

        Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2].
        The input Tensor `t` should be a gradient.
        The output will be `t` + gaussian noise.
        0.001 was said to be a good fixed value for memory networks [2].
        """
        with tf.name_scope(name, "add_gradient_noise", [t, stddev]) as name:
            t = tf.convert_to_tensor(t, name="t")
            gn = tf.random_normal(tf.shape(t), stddev=stddev)

        return tf.add(t, gn, name=name)
