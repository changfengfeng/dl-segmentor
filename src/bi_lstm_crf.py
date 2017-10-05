import tensorflow as tf
import numpy as np
import time

class Model:
    """ Define the bi direction lstm plus crf for struct prediction

    This model can be used for segmentor, pos, ner tasks of nlp
    """

    def __init__(self, vocab_size, embedding_size, hidden_size, lstm_layers,
            keep_rate, class_num, learning_rate, gradients_clip, max_seq_length
            ):
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
            #self.embeddings = tf.Variable(w2v_vectors, dtype=tf.float32,
            #    name="embeddings")

            self.inputs = tf.nn.embedding_lookup(self.embeddings, self.x_holder)

        with tf.variable_scope("lstms"):
            def _get_cell(hidden_size, keep_rate):
                lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
                cell = tf.nn.rnn_cell.DropoutWrapper(lstm,
                        output_keep_prob=keep_rate, dtype=tf.float32)
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
            #optimizer = tf.train.AdamOptimizer(self.learning_rate)
            #self.train_op = optimizer.minimize(self.loss)

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


    def do_debug(self, sess, lexicon,
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
                print(x_inputs_val)

                logits_val = sess.run(
                            self.crf_inputs,
                            {self.x_holder: x_inputs_val,
                             self.keep_rate : 1.0})
                real_lengths = np.sum(np.sign(x_inputs_val), axis=1)
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

                for s in results:
                    print("tok:", s)

    def train(self, train_inputs, validate_inputs, max_train_steps, batch_size,
            embeddings, debug_data_path, lexicon):
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

                        self.do_debug(sess, lexicon, transition_params_val,
                                debug_data_path)

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
                        print("left loss {:.4f} at step {}".format(loss_val,
                            self.train_step))


    def inference(self, x_inputs):
        pass

    def save_training_to_file(self, log_dir, model_name, step):
        pass

    def restore_training_from_file(self, log_dir, model_name, step):
        pass
