# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import time

class Model:
    """ define this model using cnn for char, concat the char to word
    embedding, the using the bi-rnn+crf to predict NER
    """

    def __init__(self, max_sentence_length, max_word_length,
            w2v_embedding, c2v_embedding, cnn_filter_height,
            cnn_filter_size, hidden_units, lstm_layers, class_num,
            learning_rate, gradients_clip, log_dir, max_train_step):

        self.max_sentence_length = max_sentence_length
        self.max_word_length = max_word_length
        self.cnn_filter_height = cnn_filter_height
        self.cnn_filter_size = cnn_filter_size
        self.hidden_units = hidden_units
        self.lstm_layers = lstm_layers
        self.class_num = class_num
        self.learning_rate = learning_rate
        self.gradients_clip = gradients_clip
        self.log_dir = log_dir
        self.max_train_step = max_train_step

        self.word_holder = tf.placeholder(dtype=tf.int32, shape=[None,
                max_sentence_length], name="input_word")
        self.char_holder = tf.placeholder(dtype=tf.int32, shape=[None,
                max_sentence_length * max_word_length], name="input_char")

        self.word_embedding = tf.Variable(w2v_embedding, dtype=tf.float32,
                name="word_embedding")
        self.char_embedding = tf.Variable(c2v_embedding, dtype=tf.float32,
                name="char_embedding")

        self.c2v_embedding_size = c2v_embedding.shape[1]

        self.cnn_filter = tf.get_variable(name="cnn_filters",
                shape=[cnn_filter_height, self.c2v_embedding_size, 1, cnn_filter_size],
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())
        self.project_w = tf.get_variable(name="project_weight",
                shape=[hidden_units * 2, class_num], dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())
        self.project_b = tf.Variable(tf.zeros(shape=[class_num], name="project_bias"))

    def inference(self, word_sequence, char_sequence, reuse):
        """ Create the graph

        Args:
            word_sequence: word input tensor
            char_sequence: char input tensor
            reuse: False for training, True for testing
        """

        word_inputs = tf.nn.embedding_lookup(self.word_embedding, word_sequence)
        char_inputs = tf.nn.embedding_lookup(self.char_embedding, char_sequence)

        with tf.variable_scope("char_cnn", reuse=reuse):
            cnn_inputs = tf.reshape(char_inputs, shape=[-1,
                self.max_word_length, self.c2v_embedding_size])
            cnn_inputs = tf.expand_dims(cnn_inputs, axis=-1)

            conv = tf.nn.conv2d(cnn_inputs, filter=self.cnn_filter,
                    strides=[1, 1, self.c2v_embedding_size, 1],
                    name="char_conv", padding="VALID")
            conv = tf.nn.relu(conv)
            max_pooling_output = tf.nn.max_pool(conv,
                ksize=[1, self.max_word_length - self.cnn_filter_height + 1, 1, 1],
                strides=[1, 1, 1, 1], padding="VALID", name="pooling")
            max_pooling_output = tf.squeeze(max_pooling_output, axis=[1,2])
            max_pooling_output = tf.reshape(max_pooling_output, shape=[-1,
                self.max_sentence_length, self.cnn_filter_size])

        with tf.variable_scope("bi-lstm", reuse=reuse):
            lstm_input = tf.concat([word_inputs, max_pooling_output], axis=-1)
            def _get_cell():
                cell = tf.nn.rnn_cell.LSTMCell(self.hidden_units, reuse=reuse)
                if not reuse:
                   cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                           output_keep_prob=0.5)
                return cell

            if self.lstm_layers > 1:
                forward_cell = tf.nn.rnn_cell.MultiRNNCell([_get_cell() for _ in range(self.lstm_layers)])
                backward_cell = tf.nn.rnn_cell.MultiRNNCell([_get_cell() for _ in range(self.lstm_layers)])
            else:
                forward_cell = _get_cell()
                backward_cell = _get_cell()

            real_sequence_length = tf.reduce_sum(tf.sign(word_sequence), axis=1)
            lstm_outputs, _ = tf.nn.bidirectional_dynamic_rnn(forward_cell,
                    backward_cell, lstm_input, dtype=tf.float32,
                    sequence_length=real_sequence_length)
            lstm_final_output = tf.concat(lstm_outputs, axis=-1)

        with tf.variable_scope("projection", reuse=reuse):
            projection_inputs = tf.reshape(lstm_final_output, shape=[-1,
                self.hidden_units * 2])
            logits = tf.matmul(projection_inputs, self.project_w) + self.project_b

        crf_input = tf.reshape(logits, shape=[-1, self.max_sentence_length,
            self.class_num], name="logits_crf" if reuse else None)
        return crf_input, real_sequence_length

    def loss(self, crf_input, real_sequence_length, y):
        crf_loss, self.transition_params = tf.contrib.crf.crf_log_likelihood(crf_input, y,
                real_sequence_length)
        loss = tf.reduce_mean(-crf_loss)
        return loss

    def train_op(self, loss):
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), self.gradients_clip)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train_op = optimizer.apply_gradients(zip(grads, tvars))
        return train_op

    def validate_op(self):
        crf_input, real_length = self.inference(self.word_holder, self.char_holder, True)
        return crf_input, real_length

    def validate(self, sess, word_sequence, char_sequence, y, transition_params, crf_input,
            real_length):
        logits, real_lengths = sess.run([crf_input, real_length],
                {self.word_holder: word_sequence, self.char_holder:
                    char_sequence})

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

    def validate_fscore(self, sess, word_sequence, char_sequence, y, transition_params, crf_input,
            real_length):
        logits, real_lengths = sess.run([crf_input, real_length],
                {self.word_holder: word_sequence, self.char_holder:
                    char_sequence})

        predict_nozero_total = 0
        true_no_zero_total = 0
        predict_match_total = 0

        for logit, validate_y, real_length in zip(
            logits, y, real_lengths):
            real_logit = logit[:real_length]
            real_validate_y = validate_y[:real_length]
            decoded_seq, _ = tf.contrib.crf.viterbi_decode(real_logit, transition_params)

            predict_nozero_total += np.sum(np.sign(decoded_seq))
            true_no_zero_total += np.sum(np.sign(real_validate_y))

            predict_match_total += np.sum(np.logical_and(np.sign(decoded_seq),
                                   np.equal(decoded_seq, real_validate_y)))

        precision = predict_match_total / predict_nozero_total
        recall = predict_match_total / true_no_zero_total
        f1 = 0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        return precision, recall, f1

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
                record_defaults=[[0] for i in range(self.max_sentence_length * (
                    self.max_word_length + 2))])
        # shuffle batches shape is [item_length, batch_size]
        shuffle_batches = tf.train.shuffle_batch(decoded,
                                  batch_size=batch_size,
                                  capacity=batch_size * 40,
                                  min_after_dequeue=batch_size)

        features = tf.transpose(tf.stack(shuffle_batches[0:self.max_sentence_length]))
        char_features = tf.transpose(tf.stack(shuffle_batches[self.max_sentence_length:
            (self.max_word_length + 1) * self.max_sentence_length]))
        labels = tf.transpose(tf.stack(shuffle_batches[self.max_sentence_length * (
            self.max_word_length + 1):]))

        train_crf_input, train_sequence_length = self.inference(features,
                char_features, False)
        loss = self.loss(train_crf_input, train_sequence_length, labels)
        train_op = self.train_op(loss)

        test_word_input, test_char_input, test_y = self.load_data(validate_data_fn)
        test_crf_input, test_real_length = self.validate_op()

        sv = tf.train.Supervisor(logdir=self.log_dir)
        with sv.managed_session(master="") as sess:
            best_accuracy = 0.0
            for step in range(self.max_train_step):
                if sv.should_stop():
                    break;
                try:
                    start = time.time()
                    loss_val, transition_params_val, _ = sess.run(
                        [loss, self.transition_params, train_op])
                    end = time.time()

                    if (step + 1) % 10 == 0:
                        print("loss {:.4f} at step {}, time {:.4f}".format(loss_val, step + 1, end-start))

                    if (step + 1) % 100 == 0 or step == 0:
                        p, r, f1 = self.validate_fscore(sess, test_word_input,
                            test_char_input, test_y, transition_params_val,
                            test_crf_input, test_real_length)
                        print("p {:.4f} r {:.4f} f1 {:.4f} at step {}".format(p,
                            r, f1, step + 1))

                        if f1 > best_accuracy:
                            best_accuracy = f1
                            sv.saver.save(sess, self.log_dir + "/best_model")
                            print("best accuracy model")
                        elif best_accuracy - f1 < 0.001:
                            sv.saver.save(sess, self.log_dir + "/best_model")
                            print("best accuracy model in margin")

                except KeyboardInterrupt as e:
                    sv.saver.save(sess, self.log_dir+'/model',
                            global_step=(step + 1))
            sv.saver.save(sess, self.log_dir + '/finnal_model')

    def load_data(self, validate_data_fn):
        wx = []
        cx = []
        y = []
        fp = open(validate_data_fn, "r")
        ln = 0
        for line in fp:
            line = line.rstrip()
            ln += 1
            if not line:
                continue
            ss = line.split(" ")
            assert (len(ss) == (self.max_sentence_length *
                            (2 + self.max_word_length)))
            lwx = []
            lcx = []
            ly = []
            for i in range(self.max_sentence_length):
                lwx.append(int(ss[i]))
                for k in range(self.max_word_length):
                    lcx.append(int(ss[self.max_sentence_length + i *
                                  self.max_word_length + k]))
                ly.append(int(ss[i + self.max_sentence_length * (
                    self.max_word_length + 1)]))
        wx.append(lwx)
        cx.append(lcx)
        y.append(ly)
        fp.close()
        return np.array(wx), np.array(cx), np.array(y)
