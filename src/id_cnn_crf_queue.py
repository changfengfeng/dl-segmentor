# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import time

class Model:
    """ Using file queue to read content and train the id-cnn-crf model
    """

    def __init__(self, w2v_embeddings, embedding_size, filter_num, filter_height,
            block_times, cnn_layers, keep_rate, class_num, learning_rate,
            max_seq_length, log_dir, max_train_steps):
        """
        Args:
            cnn_layers: [{rate:}] the sample rate for every id layer
        """
        self.filter_num = filter_num
        self.filter_height = filter_height
        self.block_times = block_times
        self.cnn_layers = cnn_layers
        self.keep_rate_scalar = keep_rate
        self.class_num = class_num
        self.learning_rate = learning_rate
        self.max_seq_length = max_seq_length
        self.log_dir = log_dir
        self.max_train_steps = max_train_steps
        self.embedding_size = embedding_size

        self.embeddings = tf.Variable(w2v_embeddings, trainable=True,
                dtype=tf.float32, name="word_embedding")

        self.projection_weight = tf.get_variable(name="projection_weight",
                shape=[block_times * filter_num, class_num],
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
        embedding_outputs = tf.nn.embedding_lookup(self.embeddings, x_holder)
        # want to keep the same length of output. so must use padding=SAME,
        # if using [height, embedding, 1, filter_size] it will padding the
        # embedding
        inputs = tf.expand_dims(embedding_outputs, axis=1)

        with tf.variable_scope("id_cnn", reuse=reuse):
            self.keep_rate = tf.get_variable(name="keep_rate",
                    shape=[], dtype=tf.float32, trainable=False,
                    initializer=tf.constant_initializer(self.keep_rate_scalar))

            filter_weight = tf.get_variable(name="first_cnn_weight",
                    shape=[1, self.filter_height, self.embedding_size, self.filter_num],
                    dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())

            id_cnn_inputs = tf.nn.conv2d(inputs, filter_weight, strides=[1, 1, 1, 1],
                    padding='SAME', name="first_conv2d")

            idcnn_final_outputs = []
            for i in range(self.block_times):
                with tf.variable_scope("idcnn_block", reuse = True if (reuse or i > 0) else False):
                    for j in range(len(self.cnn_layers)):
                        rate = self.cnn_layers[j]['rate']
                        id_cnn_weights = tf.get_variable(name="id_cnn_w%d" % j,
                                shape=[1, self.filter_height, self.filter_num, self.filter_num],
                                dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())
                        conv = tf.nn.atrous_conv2d(id_cnn_inputs,
                                id_cnn_weights,
                                rate, padding="SAME")
                        id_cnn_bias = tf.get_variable(name="id_cnn_b%d" % j,
                                shape=[self.filter_num], dtype=tf.float32)
                        conv = tf.nn.bias_add(conv, id_cnn_bias)
                        conv = tf.nn.relu(conv)

                        id_cnn_inputs = conv
                        if j + 1 == len(self.cnn_layers):
                            idcnn_final_outputs.append(conv)
            # merg the last output of echo block
            idcnn_block_outputs = tf.concat(idcnn_final_outputs, axis=-1)
            idcnn_block_outputs_dropout = tf.nn.dropout(idcnn_block_outputs, self.keep_rate)
            projection_inputs = tf.reshape(idcnn_block_outputs_dropout, [-1,
                self.block_times * self.filter_num])

        logits = tf.matmul(projection_inputs, self.projection_weight) + self.projection_bias
        crf_inputs = tf.reshape(logits, shape=[-1, self.max_seq_length,
            self.class_num], name="logits_crf" if reuse else None)
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

        test_input_x, test_input_y = self.load_data(validate_data_fn)
        test_real_length = tf.reduce_sum(tf.sign(self.test_input), axis=1)
        test_logits = self.inference(self.test_input, test_real_length,
                reuse=True)

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.learning_rate,
                        global_step, 8000, 0.9)
        train_op = tf.contrib.layers.optimize_loss(loss,
                        global_step = global_step,
                        learning_rate = learning_rate, optimizer="Adam")

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
