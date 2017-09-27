import tensorflow as tf
import numpy as np
import time

class Model:
    """ Using cnn to get the meaning of char of words
    Using bi-lstm to get the meaning of words by context
    using crf to do pos tagging or ner tagging
    """

    def __init__(self, max_sequence_length, max_word_length, w2v_embedding, c2v_embedding,
            cnn_filter_height, cnn_filter_size,
            lstm_hidden_size, lstm_layers, lstm_keep_rate, class_num,
            learning_rate, gradients_clip):
        """
        Args:
            max_sequence_length: the max input sequence length
            max_word_length: the max word length in char
            w2v_embedding: np array of word2vec embedding
            c2v_embedding: np array of char2vec embedding
            cnn_filter_height: cnn convolution height size
            cnn_filter_size: the cnn filter sizes
            lstm_hidden_size: the lstm hidden units
            lstm_layers: how many layers of bi lstm to use
            lstm_keep_rate: lstm keep rate
            class_number: the target class number
        """
        self.max_sequence_length = max_sequence_length
        self.max_word_length = max_word_length

        c2v_embedding_size = c2v_embedding.shape[1]
        w2v_embedding_size = w2v_embedding.shape[1]

        with tf.variable_scope("inputs"):
            self.x_holder = tf.placeholder(dtype=tf.int32, shape=[None,
                max_sequence_length], name="input_x")
            self.y_holder = tf.placeholder(dtype=tf.int32, shape=[None,
                max_sequence_length], name="input_y")
            self.char_holder = tf.placeholder(dtype=tf.int32, shape=[None,
                max_sequence_length * max_word_length], name="input_char")

            word_embeddings = tf.Variable(initial_value=w2v_embedding, dtype=tf.float32,
                    name="word_embeddings")
            char_embeddings = tf.Variable(initial_value=c2v_embedding,
                    dtype=tf.float32, name="char_embeddings")

            inputs_x = tf.nn.embedding_lookup(w2v_embedding, self.x_holder)
            inputs_char = tf.nn.embedding_lookup(c2v_embedding,
                    self.char_holder)
            inputs_x = tf.cast(inputs_x, tf.float32)
            inputs_char = tf.cast(inputs_char, tf.float32)

        with tf.variable_scope("char_cnn"):
            cnn_filter = tf.get_variable(name="cnn_filter_weights",
                    shape=[cnn_filter_height, c2v_embedding_size, 1,
                        cnn_filter_size], dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())
            cnn_inputs = tf.reshape(inputs_char, shape=[-1, max_word_length,
                c2v_embedding_size])
            cnn_inputs = tf.expand_dims(cnn_inputs, axis=-1)
            conv = tf.nn.conv2d(cnn_inputs, filter=cnn_filter, strides=[1, 1,
                c2v_embedding_size, 1], name="char_conv2d", padding="VALID")
            conv = tf.nn.relu(conv)
            # max pooling
            max_pooling_output = tf.nn.max_pool(conv, ksize=[1, max_word_length -
                cnn_filter_height + 1, 1, 1], strides=[1, 1, 1, 1],
                padding="VALID", name="cnn_max_pooling")
            max_pooling_output = tf.squeeze(max_pooling_output, axis=[1,2])
            max_pooling_output = tf.reshape(max_pooling_output, shape=[-1,
                max_sequence_length, cnn_filter_size])

        with tf.variable_scope("lstm"):
            lstm_input = tf.concat([inputs_x, max_pooling_output], axis=-1)

            def _get_cell(hidden_size, keep_rate):
                cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                        output_keep_prob=keep_rate)
                return cell

            self.keep_rate = tf.Variable(initial_value = lstm_keep_rate,
                    trainable=False, dtype=tf.float32, name="keep_rate")

            if lstm_layers > 1:
                forward_cell = tf.nn.rnn_cell.MultiRNNCell([_get_cell(lstm_hidden_size,
                    self.keep_rate) for _ in range(lstm_layers)])
                backward_cell = tf.nn.rnn_cell.MultiRNNCell([_get_cell(lstm_hidden_size,
                    self.keep_rate) for _ in range(lstm_layers)])
            else:
                forward_cell = _get_cell(lstm_hidden_size, self.keep_rate)
                backward_cell = _get_cell(lstm_hidden_size, self.keep_rate)

            real_sequence_length = tf.reduce_sum(tf.sign(self.x_holder), axis=1)
            lstm_outputs, _ = tf.nn.bidirectional_dynamic_rnn(forward_cell,
                    backward_cell, lstm_input, dtype=tf.float32,
                    sequence_length=real_sequence_length)
            lstm_final_output = tf.concat(lstm_outputs, axis=-1)

        with tf.variable_scope("projection"):
            project_w = tf.get_variable(name="weight", shape=[lstm_hidden_size *
                2, class_num], dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())
            project_b = tf.get_variable(name="bias", shape=[class_num],
                    dtype=tf.float32, initializer=tf.constant_initializer(0))

            projection_inputs = tf.reshape(lstm_final_output, shape=[-1,
                lstm_hidden_size * 2])

            logits = tf.matmul(projection_inputs, project_w) + project_b

        with tf.variable_scope("loss"):
            self.crf_input = tf.reshape(logits, shape=[-1, max_sequence_length,
                class_num])
            crf_loss, self.transition_params = tf.contrib.crf.crf_log_likelihood(
                    self.crf_input, self.y_holder, real_sequence_length)
            self.loss = tf.reduce_mean(-crf_loss)

            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
                    gradients_clip)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def train(self, train_inputs, validate_inputs,  max_train_steps, batch_size):
        """
        Args:
            train_input: np array, [mount, max_sequence_length +
                max_sequence_length + max_sequence_length * max_word_length]
            max_train_steps: max training steps
            batch_size: batch size
        """
        assert train_inputs.shape[1] == self.max_sequence_length * 2 + self.max_sequence_length * self.max_word_length
        assert validate_inputs.shape[1] == self.max_sequence_length * 2 + self.max_sequence_length * self.max_word_length


        batches = len(train_inputs) // batch_size;
        self.train_step = 0

        y_start_index = self.max_sequence_length * (self.max_word_length + 1)

        validate_x_inputs = validate_inputs[:, :self.max_sequence_length]
        validate_char_inputs = validate_inputs[:,self.max_sequence_length:y_start_index]
        validate_y_inputs = validate_inputs[:,y_start_index:]
        validate_real_length = np.sum(np.sign(validate_x_inputs), axis=1)

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            while self.train_step < max_train_steps:
                np.random.shuffle(train_inputs)

                for i in range(batches):
                    x_inputs = train_inputs[i * batch_size: (i + 1) * batch_size,
                            :self.max_sequence_length]
                    char_inputs = train_inputs[i * batch_size: (i + 1) * batch_size,
                            self.max_sequence_length:y_start_index]
                    y_inputs = train_inputs[i * batch_size: (i + 1) * batch_size,
                            y_start_index:]
                    keep_rate = 0.5
                    start = time.time()
                    loss_val, transition_params_val, _ = sess.run(
                            [self.loss, self.transition_params, self.train_op],
                            {self.x_holder : x_inputs,
                                self.y_holder: y_inputs,
                                self.char_holder: char_inputs,
                                self.keep_rate : keep_rate})
                    end = time.time()

                    self.train_step += 1

                    if self.train_step > max_train_steps:
                        break;
                    if self.train_step % 10 == 0:
                        print("loss {:.4f} at step {}, time {}".format(loss_val,
                            self.train_step, end - start))

                    if self.train_step % 1000 == 0:
                        logits_val = sess.run(
                            self.crf_input,
                            {self.x_holder: validate_x_inputs,
                             self.char_holder: validate_char_inputs,
                             self.keep_rate : 1.0})
                        correct_label = 0
                        total_label = 0
                        for logit, validate_y, real_length in zip(
                                logits_val, validate_y_inputs, validate_real_length):
                            real_logit = logit[:real_length]
                            real_validate_y = validate_y[:real_length]
                            decoded_seq, _ = tf.contrib.crf.viterbi_decode(real_logit, transition_params_val)
                            correct_label += np.sum(np.equal(decoded_seq,
                                real_validate_y))
                            total_label += len(decoded_seq)
                        print("validate accuracy {:.4f} on step {}".format(
                            correct_label / total_label, self.train_step))

                # train on the left inputs
                if batches * batch_size < len(train_inputs):
                    x_inputs = train_inputs[batches *
                            batch_size:,:self.max_sequence_length]
                    char_inputs = train_inputs[batches *
                            batch_size:,self.max_sequence_length:y_start_index]
                    y_inputs = train_inputs[batches * batch_size:,
                            y_start_index:]
                    keep_rate = 0.5
                    loss_val, transition_params_val, _ = sess.run(
                            [self.loss, self.transition_params, self.train_op],
                            {self.x_holder : x_inputs, self.y_holder: y_inputs,
                                self.char_holder: char_inputs,
                                self.keep_rate : keep_rate})

                    self.train_step += 1

                    if self.train_step > max_train_steps:
                        break;
                    if self.train_step % 10 == 0:
                        print("left loss {:.4f} at step {}".format(loss_val,
                            self.train_step))
