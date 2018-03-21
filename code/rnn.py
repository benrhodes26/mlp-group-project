import tensorflow as tf

class RNN:

    def __repr__(self):
        return "RNN"

    def __init__(self, dropoutPred,
		n_hidden,
		n_questions,
		maxGrad,
		maxSteps,
        encoding_dim,
        mini_batch_size):
        """Initialise task-specific parameters."""

        self.n_questions = n_questions
        self.n_hidden = n_hidden
        self.use_dropout = True #####*****
        self.max_grad = maxGrad
        self.dropoutPred = dropoutPred
        self.max_steps = maxSteps
        self.feature_len = encoding_dim
        self.mini_batch_size = mini_batch_size
        self.n_input = self.n_questions * 2

        self.build()

    def build(self):

        tf.reset_default_graph()

        # data. 'None' means any length batch size accepted
        self.inputs = tf.placeholder(
            tf.float32,
            shape=[None, self.max_steps, self.feature_len],
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

        cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_hidden)

        if self.dropoutPred & self.var_dropout:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                                 output_keep_prob=self.keep_prob,
                                                 state_keep_prob=self.keep_prob,
                                                 variational_recurrent=self.var_dropout,
                                                 dtype=tf.float32)
        else:
            # Only apply non-variational dropout to output connections
            cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                                 output_keep_prob=self.keep_prob,
                                                 dtype=tf.float32)

        initial_state = cell.zero_state(self.mini_batch_size, dtype=tf.float32)

        self.predInput, self.state = tf.nn.dynamic_rnn(cell=cell,
                                                     inputs=self.inputs,
                                                      initial_state=initial_state,
                                                     dtype=tf.float32)

        sigmoid_w = tf.get_variable(dtype=tf.float32,
                                    name="sigmoid_w",
                                    shape=[self.n_hidden,
                                           self.n_questions])
        sigmoid_b = tf.get_variable(dtype=tf.float32,
                                    name="sigmoid_b",
                                    shape=[self.n_questions])

        self.outputs = tf.reshape(self.predInput,
                                  shape=[-1, self.n_hidden])

        logits = tf.matmul(self.outputs, sigmoid_w) + sigmoid_b


        logits = tf.reshape(logits, [-1])

        self.logits = tf.dynamic_partition(logits, self.target_ids, 2)[1]

        loss_per_example = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.logits, labels=self.targets)
        self.loss = tf.reduce_mean(loss_per_example)
        #####To Here######
        self.summary_loss = [tf.summary.scalar('loss', self.loss)]
        # track number of batches seen
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        self.learning_rate = tf.placeholder_with_default(1.0, shape=(),
                                                         name='learning_rate')

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)




    def calcGrad(self, batch, rate, alpha):
        n_steps = batch.max_num_ans
        n_student = batch.batch_size
        if(n_steps > self.max_steps):
            print(n_steps, self.max_steps)

        maxNorm = 0
