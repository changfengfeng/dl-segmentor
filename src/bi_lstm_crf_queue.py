import tensorflow as tf
import numpy as np
import time

class Model:
    """ Define model with different inputs.

    Training using the tf queue to do batch inptus
    Inference using the numpy array to do batch inputs
    """

    def __init__(self, w2v_embeddings, hidden_units,
            lstm_layers, class_num, learning_rate, keep_rate,
            gradients_clip, max_seq_length, log_dir, max_train_steps):

        self.hidden_units = hidden_units;
        self.lstm_layers = lstm_layers
        self.class_num = class_num
        self.learning_rate = learning_rate
        self.keep_rate_scalar = keep_rate
        self.gradients_clip = gradients_clip
        self.max_seq_length = max_seq_length
        self.log_dir = log_dir
        self.max_train_steps = max_train_steps

        self.embeddings = tf.Variable(w2v_embeddings, trainable=True,
                dtype=tf.float32, name="word_embedding")

        self.projection_weight = tf.get_variable(name="projection_weight",
                shape=[hidden_units * 2, class_num],
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())

        self.projection_bias = tf.get_variable(name="projection_bias",
                shape=[class_num],
                dtype=tf.float32,
                initializer=tf.constant_initializer(0))
        self.test_input = tf.placeholder(name="test_input_x",
                shape=[None, max_seq_length], dtype=tf.int32)

    def inference(self, x_holder, real_length, reuse):
        """ Create the graph for inference and test

        Args:
            x_holder: the inputs tensor [batch_size, max_seq_length]
            real_length: the real length of input [batch_size]
            reuse: True for the testing, False for the training

        Returns:
            return the logits tensor
        """
        inputs = tf.nn.embedding_lookup(self.embeddings, x_holder)

        with tf.variable_scope("lstm", reuse=reuse):
            self.keep_rate = tf.get_variable(name="keep_rate",
                    shape=[], dtype=tf.float32, trainable=False,
                    initializer=tf.constant_initializer(self.keep_rate_scalar))

            def __get_cell(hidden_size, keep_rate):
                lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_size,
                        reuse=reuse)
                cell = tf.nn.rnn_cell.DropoutWrapper(lstm,
                        output_keep_prob=keep_rate, dtype=tf.float32)
                return cell

            if self.lstm_layers > 1:
                forward_lstm_cell = tf.nn.rnn_cell.MultiRNNCell(
                        [__get_cell(self.hidden_size, self.keep_rate) for _ in self.lstm_layers])
                backward_lstm_cell = tf.nn.rnn_cell.MultiRNNCell(
                        [__get_cell(self.hidden_units, self.keep_rate) for _ in self.lstm_layers])
            else:
                forward_lstm_cell = __get_cell(self.hidden_units,
                        self.keep_rate)
                backward_lstm_cell = __get_cell(self.hidden_units,
                        self.keep_rate)
            lstm_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                    forward_lstm_cell, backward_lstm_cell,
                    inputs, real_length, dtype=tf.float32)
            bi_outputs = tf.concat(lstm_outputs, axis=2)

            projection_inputs = tf.reshape(bi_outputs, shape=[-1,
                self.hidden_units * 2])
            logits = tf.matmul(projection_inputs,
                    self.projection_weight) + self.projection_bias
            crf_inputs = tf.reshape(logits, shape=[-1, self.max_seq_length,
                self.class_num])
            return crf_inputs

    def loss(self, x_holder, y_holder):
        """ Compute the loss for the training

        Args:
            x_holder: input tenor [batch_size, max_seq_length]
            y_holder: input tensor [batch_size, max_seq_length]

        Returns:
            return the loss tensor
        """
        real_length = tf.reduce_sum(tf.sign(x_holder), axis=1)
        crf_inputs = self.inference(x_holder, real_length, False)

        loss, self.transition_params = tf.contrib.crf.crf_log_likelihood(
                    crf_inputs, y_holder, real_length)
        loss = tf.reduce_mean(-loss)
        return loss

    def calculate_accuracy(self, logits, y, real_lengths, transition_params):
        """ get the accuracy of crf tagging

        Args:
            logits: the computed logits val [batch_size, max_seq_length,
            class_num]
            y: the real label [batch_size, max_seq_length]
            real_lengths: the real length
            transition_params: crf transition matrix

        Returns:
            Accuray float32
        """

        correct_label = 0
        total_label = 0

        for logit, validate_y, real_length in zip(
            logits, y, real_lengths):
            real_logit = logit[:real_length]
            real_validate_y = validate_y[:real_length]
            decoded_seq, _ = tf.contrib.crf.viterbi_decode(real_logit, transition_params)
            correct_label += np.sum(np.equal(decoded_seq, real_validate_y))
            total_label += len(decoded_seq)

        return correct_label / total_label

    def load_data(self, validate_data_fn):
        """ Read the validate data

        Args:
            validate_data_fn: validate data file path

        Returns:
            (x, y) numpy arrays
        """
        x_inputs = []
        y_inputs = []
        with open(validate_data_fn, "r") as f:
            while True:
                line = f.readline()
                line = line.strip()
                if line == None or len(line) == 0:
                    break
                ints =  np.array(line.split(" "))
                ints = ints.astype(int)
                assert len(ints) == self.max_seq_length * 2
                x_inputs.append(ints[0:self.max_seq_length])
                y_inputs.append(ints[self.max_seq_length:])

        print("reading {} lines validate data".format(len(x_inputs)))
        assert len(x_inputs) == len(y_inputs)
        return np.array(x_inputs, dtype="int32"), np.array(y_inputs, dtype="int32")

    def train(self, batch_size, train_data_fn, validate_data_fn):
        """ Creat train op to train the graph on the data

        Args:
            batch_size: batch size
            train_data_fn: the train data file name
            validate_data_fn: the validate data fn
        """

        filename_queue = tf.train.string_input_producer([train_data_fn])
        reader = tf.TextLineReader(skip_header_lines=0)
        key, value = reader.read(filename_queue)
        decoded = tf.decode_csv(
                value,
                field_delim=' ',
                record_defaults=[[0] for i in range(self.max_seq_length * 2)])
        # shuffle batches shape is [item_length, batch_size]
        shuffle_batches = tf.train.shuffle_batch(decoded,
                                  batch_size=batch_size,
                                  capacity=batch_size * 50,
                                  min_after_dequeue=batch_size)

        features = tf.transpose(tf.stack(shuffle_batches[0:self.max_seq_length]))
        labels = tf.transpose(tf.stack(shuffle_batches[self.max_seq_length:]))

        loss = self.loss(features, labels)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),
                    self.gradients_clip)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train_op = optimizer.apply_gradients(zip(grads, tvars))

        test_input_x, test_input_y = self.load_data(validate_data_fn)
        test_real_length = tf.reduce_sum(tf.sign(self.test_input), axis=1)
        test_logits = self.inference(self.test_input, test_real_length,
                reuse=True)

        sv = tf.train.Supervisor(logdir=self.log_dir)
        with sv.managed_session(master="") as sess:
            best_accuracy = 0.0
            for step in range(self.max_train_steps):
                if sv.should_stop():
                    break;
                keep_rate = 0.5
                start = time.time()
                loss_val, transition_params_val, _ = sess.run(
                        [loss, self.transition_params, train_op],
                        {self.keep_rate : keep_rate})
                end = time.time()

                if step > 0 and step % 10 == 0:
                    print("loss {:.4f} at step {}, time {}".format(loss_val, step, end-start))

                if step > 0 and step % 1000 == 0:
                    logits, test_real_length_val = sess.run(
                            [test_logits, test_real_length],
                            {self.test_input: test_input_x,
                             self.keep_rate: 1.0})

                    accuracy = self.calculate_accuracy(logits, test_input_y,
                            test_real_length_val, transition_params_val)
                    print("accuracy {:.4f} at step {}".format(accuracy, step))
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        sv.saver.save(sess, self.log_dir + "/best_model")
                        print("best accuracy model")
