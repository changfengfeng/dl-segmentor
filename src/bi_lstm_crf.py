import tensorflow as tf
import numpy as np

class Model:
    """ Define the bi direction lstm plus crf for struct prediction

    This model can be used for segmentor, pos, ner tasks of nlp
    """

    def __init__(self, vocab_size, embedding_size, hidden_size, lstm_layers,
            keep_rate, class_num, learning_rate, gradients_clip, max_seq_length):
        """

        Args:
            vocab_size: vocabulary size for embedding layer
            embedding_size: embedding size for embedding layer
            hidden_size: hidden units size of lstm layer
            lstm_layers: if using multilayers lstm
            keep_rate: the keep rate of every lstm output
            learning_rate: the initial learning rate
            gradients_clip: using gradients clip for lstm training
        """

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.learning_rate = learning_rate
        self.gradients_clip = gradients_clip
        self.class_num = class_num
        self.max_seq_length = max_seq_length

        # begin to create the graph

        with tf.variable_scope("inputs"):
            # the input is [batch_size, max_sequence_steps]
            self.x_holder = tf.placeholder(tf.int32, shape=[None, max_seq_length],
                    name="input_x")
            self.y_holder = tf.placeholder(tf.int32, shape=[None, max_seq_length],
                    name="input_x")

            # using x to get the real length, 0 value stands for </s>
            self.real_length = tf.reduce_sum(tf.sign(self.x_holder), axis=1)

            self.keep_rate = tf.Variable(keep_rate, trainable=False,
                    name="keep_rate")

            # initialize from the pretrained word2vec model
            self.embeddings = tf.get_variable(name="embeddings",
                    shape=[self.vocab_size, self.embedding_size],
                    dtype=tf.float32)

            self.inputs = tf.nn.embedding_lookup(self.embeddings, self.x_holder)

        with tf.variable_scope("lstms"):
            def _get_cell(hidden_size, keep_rate):
                lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
                cell = tf.nn.rnn_cell.DropoutWrapper(lstm,
                        output_keep_prob=self.keep_rate)
                return cell

            if self.lstm_layers > 1:
                forward_lstm_cell = tf.nn.rnn_cell.MultiRNNCell(
                        [_get_cell(self.hidden_size, self.keep_rate) for _ in
                            self.lstm_layers])
                backward_lstm_cell = tf.nn.rnn_cell.MultiRNNCell(
                        [_get_cell(self.hidden_size, self.keep_rate) for _ in
                            self.lstm_layers])
            else:
                forward_lstm_cell = _get_cell(self.hidden_size, self.keep_rate)
                backward_lstm_cell = _get_cell(self.hidden_size, self.keep_rate)

            lstm_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                    forward_lstm_cell, backward_lstm_cell,
                    self.inputs, self.real_length, dtype=tf.float32)
            self.bi_outputs = tf.concat(lstm_outputs, axis=2)

        with tf.variable_scope("projection"):
            w = tf.get_variable(name="w", shape=[hidden_size * 2, class_num],
                    dtype=tf.float32,
                    initializer = tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name="bias", shape=[class_num],
                    dtype=tf.float32,
                    initializer = tf.constant_initializer(0))
            projection_inputs = tf.reshape(self.bi_outputs, shape=[-1, hidden_size * 2])
            self.logits = tf.matmul(projection_inputs, w) + b

        with tf.variable_scope("loss"):
            self.crf_inputs = tf.reshape(self.logits, shape=[-1, max_seq_length,
                class_num])
            loss, self.transition_params = tf.contrib.crf.crf_log_likelihood(
                    self.crf_inputs, self.y_holder, self.real_length)
            self.loss = tf.reduce_mean(-loss)

            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
                    gradients_clip)
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def train(self, train_inputs, validate_inputs, max_train_steps, batch_size, embeddings):
        """ Train the model.

        Args:
            train_inputs: shape=[whole_length, max_seq_length * 2] numpy array
                          concat x and y
            validate_inputs: shape=[whole_length, max_seq_length * 2] numpy array
                             concat x and y
            max_train_steps: the max training steps
            batch_size: batch size for training
        """
        assert train_inputs.shape[1] == self.max_seq_length * 2
        assert validate_inputs.shape[1] == self.max_seq_length * 2

        batches = len(train_inputs) // batch_size;
        self.train_step = 0

        validate_x_inputs = validate_inputs[:, :self.max_seq_length]
        validate_y_inputs = validate_inputs[:, self.max_seq_length:]
        validate_real_length = np.sum(np.sign(validate_x_inputs), axis=1)

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            # asign the pretrained char embedding
            embedding_input = tf.constant(np.array(embeddings), dtype=tf.float32)
            assign_embedding_op = tf.assign(self.embeddings, embedding_input)
            embedding_in_graph = sess.run(assign_embedding_op);

            while self.train_step < max_train_steps:
                np.random.shuffle(train_inputs)

                for i in range(batches):
                    x_inputs = train_inputs[i * batch_size: (i + 1) * batch_size,
                            :self.max_seq_length]
                    y_inputs = train_inputs[i * batch_size: (i + 1) * batch_size,
                            self.max_seq_length:]
                    keep_rate = 0.5
                    loss_val, transition_params_val, _ = sess.run(
                            [self.loss, self.transition_params, self.train_op],
                            {self.x_holder : x_inputs, self.y_holder: y_inputs,
                                self.keep_rate : keep_rate})

                    self.train_step += 1

                    if self.train_step > max_train_steps:
                        break;
                    if self.train_step % 10 == 0:
                        print("loss {:.4f} at step {}".format(loss_val,
                            self.train_step))

                    if self.train_step % 100 == 0:
                        logits_val = sess.run(
                            [self.crf_inputs],
                            {self.x_holder: validate_x_inputs,
                             self.y_holder: validate_y_inputs,
                             self.keep_rate : 1.0})

                        correct_label = 0
                        total_label = 0
                        for logit, validate_y, real_length in zip(
                                logits_val, validate_y_inputs, validate_real_length):
                            real_logit = logit[:real_length]
                            real_validate_y = validate_y[:real_length]
                            print(real_logit.shape, real_validate_y.shape)
                            decoded_seq = tf.contrib.crf.viterbi_decode(real_logit, transition_params_val)
                            correct_label += np.sum(np.equal(decoded_seq,
                                real_validate_y))
                            total_label += len(decoded_seq)
                        print("validate accuracy {:.4f} on step {}".format(
                            correct_label / total_label, self.train_step))

                # train on the left inputs
                if batches * batch_size < len(train_inputs):
                    x_inputs = train_inputs[batches * batch_size:,:self.max_seq_length]
                    y_inputs = train_inputs[batches * batch_szie:,self.max_seq_length:]
                    keep_rate = 0.5
                    loss_val, transition_params_val, _ = sess.run(
                            [self.loss, self.transition_params, self.train_op],
                            {self.x_holder : x_inputs, self.y_holder: y_inputs,
                                self.keep_rate : keep_rate})

                    self.train_step += 1

                    if self.train_step > max_train_steps:
                        break;
                    if self.train_step % 10 == 0:
                        print("loss {:.4f} at step {}".format(loss_val,
                            self.train_step))


    def inference(self, x_inputs):
        pass

    def save_training_to_file(self, log_dir, model_name, step):
        pass

    def restore_training_from_file(self, log_dir, model_name, step):
        pass
