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

        self.embeddings = tf.Variable(w2v_embeddings, name="word_embedding", dtype=tf.float32)

        self.projection_weight = tf.get_variable(name="projection_weight",
                shape=[hidden_units * 2, class_num],
                initializer=tf.contrib.layers.xavier_initializer(),
                regularizer=tf.contrib.layers.l2_regularizer(0.001))

        self.projection_bias = tf.Variable(tf.zeros([class_num],
            name="projection_bias"))

        self.test_input = tf.placeholder(name="test_input_x",
                shape=[None, max_seq_length], dtype=tf.int32)

        # for debug display
        self.pair_marks = {"(": ")",
                "（" : "）",
                "["  : "]",
                "【" : "】",
                "《" : "》",
                '“'  : '”'}

        self.break_marks = set(["。", ",", "，", " ", "\t", "?", "？", "!",
            "！", ";", "；"])
        self.pair_marks_reverse = {v:k for k, v in self.pair_marks.items()}

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

        length_64 = tf.cast(real_length, tf.int64)
        with tf.variable_scope("bilstm", reuse=reuse):
            forward_output, _ = tf.nn.dynamic_rnn(
                tf.contrib.rnn.LSTMCell(self.hidden_units,
                                        reuse=reuse),
                inputs,
                dtype=tf.float32,
                sequence_length=real_length,
                scope="RNN_forward")
            backward_output_, _ = tf.nn.dynamic_rnn(
                tf.contrib.rnn.LSTMCell(self.hidden_units,
                                        reuse=reuse),
                inputs=tf.reverse_sequence(inputs,
                                           length_64,
                                           seq_dim=1),
                dtype=tf.float32,
                sequence_length=real_length,
                scope="RNN_backword")

        backward_output = tf.reverse_sequence(backward_output_,
                                              length_64,
                                              seq_dim=1)

        output = tf.concat([forward_output, backward_output], 2)
        output = tf.reshape(output, [-1, self.hidden_units * 2])
        if reuse is None or not reuse:
            output = tf.nn.dropout(output, 0.5)

        matricized_unary_scores = tf.matmul(output, self.projection_weight) + self.projection_bias
        unary_scores = tf.reshape(
            matricized_unary_scores,
            [-1, self.max_seq_length, self.class_num],
            name="Reshape_7" if reuse else None)
        return unary_scores

    def loss(self, x_holder, y_holder):
        """ Compute the loss for the training

        Args:
            x_holder: input tenor [batch_size, max_seq_length]
            y_holder: input tensor [batch_size, max_seq_length]

        Returns:
            return the loss tensor
        """
        real_length = tf.reduce_sum(tf.sign(tf.abs(x_holder)), axis=1)
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

    def split_text(self, sentence):
        """ split sentence to sentences by colon"""
        results = []
        start = 0
        # state = 0 in pair marks, state = 1 not in pair makrs
        state = 0
        for idx in range(len(sentence)):
            char = sentence[idx]
            if char in self.pair_marks:
                state = 1
                if start < idx: # for case [][]
                    results.append(sentence[start:idx])
                start = idx
            elif char in self.pair_marks_reverse and state == 1:
                if self.pair_marks_reverse[char] == sentence[start]:
                    state = 0
                    results.append(sentence[start:idx+1])
                    start = idx + 1
            elif char in self.break_marks:
                results.append(sentence[start:idx+1])
                start = idx + 1
        if start < len(sentence):
            results.append(sentence[start:])
        outputs = []
        for i in range(len(results)):
            r = results[i].strip()
            if len(r) > 0:
                outputs.append(r)
        return outputs


    def do_debug(self, logits, test_real_length, sess, lexicon,
            transition_params, debug_data_path):
        """ Read the debug file, do inference, and output the segmented results

        Args:
            logits: crf inputs tensor
            test_real_length: real length tensor
            transition_params: the transition maxtrix
        """
        with open(debug_data_path, "r") as f:
            for line in f:
                sentence = line.strip()
                split_sentences = self.split_text(sentence)

                x_inputs_val = np.zeros([len(split_sentences), self.max_seq_length], dtype="int32")
                for i in range(len(split_sentences)):
                    for j in range(len(split_sentences[i])):
                        s = split_sentences[i][j]
                        sid = lexicon[s] if s in lexicon else lexicon['<UNK>']
                        x_inputs_val[i,j] = sid
                print(split_sentences)
                logits_val, real_lengths = sess.run(
                        [logits, test_real_length],
                        {self.test_input: x_inputs_val})
                decoded_results = []
                for logit, real_length, sentence in zip(logits_val, real_lengths,
                    split_sentences):
                    real_logit = logit[:real_length]
                    decoded_seq, _ = tf.contrib.crf.viterbi_decode(real_logit,
                            transition_params)
                    decoded_results.append(decoded_seq)

                # decode crf
                results = []
                for sentence, tags in zip(split_sentences, decoded_results):
                    assert len(sentence) == len(tags)
                    word = ""
                    for s, t in zip(sentence, tags):
                        print(s, t)
                        if t == 0:
                            results.append(s)
                        if t == 1:
                            if len(word) > 0:
                                results.append(word)
                            word = s
                        if t == 2:
                            word += s
                        if t == 3:
                            word += s
                            results.append(word)
                            word = ""
                print("]  tok:[".join(results))

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

    def train(self, batch_size, train_data_fn, validate_data_fn,
            debug_data_path, lexicon):
        """ Creat train op to train the graph on the data

        Args:
            batch_size: batch size
            train_data_fn: the train data file name
            validate_data_fn: the validate data fn
        """
        print("training ", train_data_fn)
        print("validate ", validate_data_fn)

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
        #tvars = tf.trainable_variables()
        #grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),
        #            self.gradients_clip)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train_op = optimizer.minimize(loss)

        test_input_x, test_input_y = self.load_data(validate_data_fn)
        test_real_length = tf.reduce_sum(tf.sign(tf.abs(self.test_input)), axis=1)
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
                        {})
                end = time.time()

                if step > 0 and step % 10 == 0:
                    print("loss {:.4f} at step {}, time {}".format(loss_val, step, end-start))

                    self.do_debug(test_logits, test_real_length, sess, lexicon,
                            transition_params_val, debug_data_path)

                if step > 0 and step % 1000 == 0:
                    logits, test_real_length_val = sess.run(
                            [test_logits, test_real_length],
                            {self.test_input: test_input_x,
                                })

                    accuracy = self.calculate_accuracy(logits, test_input_y,
                            test_real_length_val, transition_params_val)
                    print("accuracy {:.4f} at step {}".format(accuracy, step))
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        sv.saver.save(sess, self.log_dir + "/best_model")
                        print("best accuracy model")
