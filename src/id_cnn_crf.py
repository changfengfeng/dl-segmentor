import tensorflow as tf
import numpy as np
import time

class Model:
    """
    Define the id cnn + crf model for sequence tagging
    """

    def __init__(self, vocab_size, embedding_size, filter_num, filter_height,
            block_times, cnn_layers, keep_rate, class_num, learning_rate,
            max_seq_length):
        """
        Args:
            vocab_size: vocabulary size for embedding layer
            embedding_size: embedding size for embedding layer
            filter_size: kernel size of cnn
            block_times: id cnn block repeat times
            cnn_layers: the cnn layers config list
            keep_rate: the keep rate of every lstm output
            learning_rate: the initial learning rate
            max_seq_len: the max sequence length
        """

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.class_num = class_num
        self.max_seq_length = max_seq_length
        self.block_times = block_times
        self.cnn_layers = cnn_layers
        self.filter_num = filter_num
        self.filter_height = filter_height

        # begin to create the graph

        with tf.variable_scope("inputs"):
            # the input is [batch_size, max_sequence_steps]
            self.x_holder = tf.placeholder(tf.int32, shape=[None, max_seq_length],
                    name="input_x")
            self.y_holder = tf.placeholder(tf.int32, shape=[None, max_seq_length],
                    name="input_y")

            # using x to get the real length, 0 value stands for </s>
            self.real_length = tf.reduce_sum(tf.sign(self.x_holder), axis=1)

            self.keep_rate = tf.Variable(keep_rate, trainable=False,
                    name="keep_rate", dtype=tf.float32)

            # initialize from the pretrained word2vec model
            self.embeddings = tf.get_variable(name="embeddings",
                    shape=[self.vocab_size, self.embedding_size],
                    dtype=tf.float32,
                    initializer=tf.truncated_normal_initializer(stddev=0.1))

            embedding_outputs = tf.nn.embedding_lookup(self.embeddings, self.x_holder)
            # shape is (?,1,max_seq,embedding)
            self.inputs = tf.expand_dims(embedding_outputs, axis=1)

        with tf.variable_scope("idcnn"):
            filter_weight = tf.get_variable(name="first_cnn_weight",
                    shape=[1, filter_height, embedding_size, filter_num],
                    dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())
            id_cnn_inputs = tf.nn.conv2d(self.inputs,
                    filter_weight, strides=[1, 1, 1, 1], padding='SAME')
            idcnn_final_outputs = []
            for i in range(block_times):
                with tf.variable_scope("idcnn_block", reuse = True if i > 0 else
                        False):
                    for j in range(len(cnn_layers)):
                        rate = cnn_layers[j]['rate']
                        id_cnn_weights = tf.get_variable(name="id_cnn_w%d" % j,
                                shape=[1, filter_height, filter_num, filter_num],
                                dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())
                        conv = tf.nn.atrous_conv2d(id_cnn_inputs,
                                id_cnn_weights,
                                rate, padding="SAME")
                        id_cnn_bias = tf.get_variable(name="id_cnn_b%d" % j,
                                shape=[filter_num], dtype=tf.float32)
                        conv = tf.nn.bias_add(conv, id_cnn_bias)
                        conv = tf.nn.relu(conv)

                        id_cnn_inputs = conv
                        if j + 1 == len(cnn_layers):
                            idcnn_final_outputs.append(conv)
            # merg the last output of echo block
            idcnn_block_outputs = tf.concat(idcnn_final_outputs, axis=-1)
            idcnn_block_outputs_dropout = tf.nn.dropout(idcnn_block_outputs,
                    self.keep_rate)
            projection_inputs = tf.reshape(idcnn_block_outputs, [-1, block_times * filter_num])

        with tf.variable_scope("projection"):
            w = tf.get_variable(name="w", shape=[block_times * filter_num, class_num],
                    dtype=tf.float32,
                    initializer = tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name="bias", shape=[class_num],
                    dtype=tf.float32,
                    initializer = tf.constant_initializer(0))
            self.logits = tf.matmul(projection_inputs, w) + b

        with tf.variable_scope("loss"):
            self.crf_inputs = tf.reshape(self.logits, shape=[-1, max_seq_length,
                class_num])
            loss, self.transition_params = tf.contrib.crf.crf_log_likelihood(
                    self.crf_inputs, self.y_holder, self.real_length)
            self.loss = tf.reduce_mean(-loss)

            # TODO
            self.global_step = tf.Variable(0, trainable=False)
            self.learning_rate = tf.train.exponential_decay(learning_rate,
                        self.global_step, 4000, 0.9)
            self.train_op = tf.contrib.layers.optimize_loss(self.loss,
                        global_step = self.global_step,
                        learning_rate = self.learning_rate, optimizer="Adam")

            #optimizer = tf.train.AdamOptimizer(self.learning_rate)
            #self.train_op = optimizer.minimize(self.loss)

    def train(self, train_inputs, validate_inputs, max_train_steps, batch_size,
            embeddings):
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
            embedding_input = tf.constant(np.array(embeddings), dtype=tf.float32)
            assign_embedding_op = tf.assign(self.embeddings, embedding_input)
            sess.run(assign_embedding_op);

            while self.train_step < max_train_steps:
                np.random.shuffle(train_inputs)

                for i in range(batches):
                    x_inputs = train_inputs[i * batch_size: (i + 1) * batch_size,
                            :self.max_seq_length]
                    y_inputs = train_inputs[i * batch_size: (i + 1) * batch_size,
                            self.max_seq_length:]
                    keep_rate = 0.5
                    start = time.time()
                    loss_val, transition_params_val, _ = sess.run(
                            [self.loss, self.transition_params, self.train_op],
                            {self.x_holder : x_inputs, self.y_holder: y_inputs,
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
                            self.crf_inputs,
                            {self.x_holder: validate_x_inputs,
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
                    x_inputs = train_inputs[batches * batch_size:,:self.max_seq_length]
                    y_inputs = train_inputs[batches * batch_size:,self.max_seq_length:]
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
