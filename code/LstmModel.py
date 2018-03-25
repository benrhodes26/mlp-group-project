import tensorflow as tf


class LstmModel:

    def __repr__(self):
        return "LstmModel"

    def __init__(self, max_time_steps=973, feature_len=293,
                 n_distinct_questions=146, var_dropout=True, batch_size=32):
        """Initialise task-specific parameters."""
        self.max_time_steps = max_time_steps
        self.feature_len = feature_len
        self.n_distinct_questions = n_distinct_questions
        self.var_dropout = var_dropout
        self.acc_init = None
        self.auc_init = None
        self.summary_loss = None
        self.summary_aucacc = None
        self.batch_size = batch_size
        self.reuse = False

    def build_graph(self, n_hidden_units=200, clip_norm=5*1e-5, optimisation='adam'):
        self._build_model(n_hidden_units=n_hidden_units)
        self._build_training(clip_norm=clip_norm, optimisation=optimisation)
        self._build_metrics()

    def _build_model(self, n_hidden_units=200):
        """Build a TensorFlow computational graph for an LSTM network.

        Model based on "DKT paper" (see section 3):
            Piech, Chris, et al. "Deep knowledge tracing."
            Advances in Neural Information Processing Systems. 2015.

        Implementation based on "GD paper" (see section 3):
            Xiong, Xiaolu, et al. "Going Deeper with Deep Knowledge Tracing."
            EDM. 2016.

        Parameters
        ----------
        n_hidden_units : int (default=200)
            200 hidden units were used in DKT paper
        is_training: bool (default=True)
            if False, we are evaluating on validation set, so reuse
            RNN parameters from training phase
        """

        self.keep_prob = tf.placeholder_with_default(1.0, shape=(),
                                                     name='keep_prob')

        with tf.variable_scope('RNN', reuse=self.reuse,
                               initializer=tf.random_uniform_initializer(-0.05, 0.05)):

            cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units)
            if self.var_dropout:
                # Apply variational dropout to recurrent state and output
                cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                                     output_keep_prob=self.keep_prob,
                                                     state_keep_prob=self.keep_prob,
                                                     variational_recurrent=self.var_dropout,
                                                     dtype=tf.float32)
            else:
                # Apply non-variational dropout to output
                cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                                     output_keep_prob=self.keep_prob,
                                                     dtype=tf.float32)

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

            # make first dim batch_size times max_time_steps
            self.outputs = tf.reshape(self.outputs,
                                      shape=[-1, n_hidden_units])

            logits = tf.matmul(self.outputs, sigmoid_w) + sigmoid_b
            logits = tf.reshape(logits, [-1])
            self.logits = tf.dynamic_partition(logits, self.target_ids, 2)[1]

            loss_per_example = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.logits, labels=self.targets)
            self.loss = tf.reduce_mean(loss_per_example)
            self.summary_loss = [tf.summary.scalar('loss', self.loss)]

            # need predictions to calculate accuracy and auc
            self.predictions = tf.nn.sigmoid(self.logits)

    def _build_training(self, clip_norm=5*1e-5, optimisation='adam'):
        """Define parameters updates."""

        # track number of batches seen
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        self.learning_rate = tf.placeholder_with_default(1.0, shape=(),
                                                         name='learning_rate')

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Ensures that we execute the update_ops before performing the train_step

            if optimisation == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            elif optimisation == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
            elif optimisation == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.98)
            elif optimisation == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

            grads, trainable_vars = list(zip(*optimizer.compute_gradients(self.loss)))

            if clip_norm:
                # grads, _ = tf.clip_by_global_norm(grads, clip_norm)
                grads = [tf.clip_by_norm(grad, clip_norm) for grad in grads]

            self.grads_and_vars = list(zip(grads, trainable_vars))
            self.training = optimizer.apply_gradients(
                self.grads_and_vars,
                global_step=self.global_step)

    def _build_metrics(self):
        """Compute accuracy and AUC."""
        self.accuracy = tf.metrics.accuracy(labels=self.targets,
                                            predictions=tf.round(self.predictions),
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
       
