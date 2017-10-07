# -*- coding:utf-8 -*-

import tensorflow as tf

class Model:
    def __init__(self,
                 numHidden,
                 maxSeqLen,
                 numTags):
        self.num_hidden = numHidden
        self.num_tags = numTags
        self.max_seq_len = maxSeqLen
        self.W = tf.get_variable(
            shape=[numHidden * 2, numTags],
            initializer=tf.contrib.layers.xavier_initializer(),
            name="weights",
            regularizer=tf.contrib.layers.l2_regularizer(0.001))
        self.b = tf.Variable(tf.zeros([numTags], name="bias"))

    def inference(self, X, length, reuse=False):
        length_64 = tf.cast(length, tf.int64)
        with tf.variable_scope("bilstm", reuse=reuse):
            def _get_cell(num_hidden, keep_rate):
                cell = tf.nn.rnn_cell.LSTMCell(num_hidden, reuse=reuse)
                if not reuse:
                    cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                            output_keep_prob=keep_rate)
                return cell
            bilstm_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                    _get_cell(self.num_hidden, 0.5),
                    _get_cell(self.num_hidden, 0.3),
                    X, length, dtype=tf.float32)

        output = tf.concat(bilstm_outputs, 2)
        output = tf.reshape(output, [-1, self.num_hidden * 2])

        matricized_unary_scores = tf.matmul(output, self.W) + self.b
        unary_scores = tf.reshape(
            matricized_unary_scores,
            [-1, self.max_seq_len, self.num_tags],
            name="Reshape_7" if reuse else None)
        return unary_scores
