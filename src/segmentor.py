import tensorflow as tf
import numpy as np
import word2vec as w2v
import time

class Segmentor:
    """ Import lexicon and model, segment text
    """

    def __init__(self, lexicon_path, model_path, model_prefix, max_seq_length):

        self.max_seq_length = max_seq_length
        self.lexicon_path = lexicon_path
        self.model_path = model_path
        self.model_prefix = model_prefix

        # load lexicon
        model = w2v.load(lexicon_path)
        self.lexicon = {w:i for i, w in enumerate(model.vocab)}
        self.vocab = model.vocab

        # load model
        with tf.gfile.GFile(model_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        tf.import_graph_def(graph_def, input_map=None, return_elements=None,
            name=model_prefix, op_dict=None, producer_op_list=None)

        graph = tf.get_default_graph()
        self.x_inputs = graph.get_tensor_by_name("%s/test_input_x:0" % model_prefix)
        self.keep_rate = graph.get_tensor_by_name("%s/lstm/keep_rate:0" % model_prefix)
        self.logits = graph.get_tensor_by_name("%s/logits_crf:0" % model_prefix)
        self.transition_params = graph.get_tensor_by_name("%s/transitions:0" %
                model_prefix)
        self.sess = tf.Session()

        print(self.transition_params)
        print(self.logits)

    def test_accuracy(self, validate_data_fn):
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
        x_inputs_array = np.array(x_inputs, dtype="int32")
        y_inputs_array = np.array(y_inputs, dtype="int32")

        test_real_length = tf.reduce_sum(tf.sign(self.x_inputs), axis=1)

        logits_val, keep_rate_val, transitions, real_lengths = self.sess.run(
                [self.logits, self.keep_rate, self.transition_params,
                    test_real_length],
                {self.x_inputs: x_inputs_array, self.keep_rate: 1.0})

        correct_label = 0
        total_label = 0

        for logit, validate_y, real_length in zip(
            logits_val, y_inputs_array, real_lengths):
            real_logit = logit[:real_length]
            real_validate_y = validate_y[:real_length]
            decoded_seq, _ = tf.contrib.crf.viterbi_decode(real_logit, transitions)
            correct_label += np.sum(np.equal(decoded_seq, real_validate_y))
            total_label += len(decoded_seq)

        print("validate accuray: {:.4f}".format(correct_label / total_label))

    def segment(self, sentence):
        """segment sentence

        Args:
            sentence: the input string
        Returns:
            list of string
        """
        inputs = []
        for s in sentence:
            if s in [" ", "\t", "\n", "\r"]:
                continue
            sid = self.lexicon[s] if s in self.lexicon else self.lexicon['<UNK>']
            inputs.append(sid)
        real_length = len(inputs)
        for i in range(len(inputs), 80):
            inputs.append(0)

        x_inputs_val = [inputs]
        start = time.time()
        logits_val, keep_rate_val, transitions = self.sess.run(
                [self.logits, self.keep_rate, self.transition_params],
                {self.x_inputs: x_inputs_val, self.keep_rate: 1.0})
        real_logit = logits_val[0][:real_length]
        decoded_seq, _ = tf.contrib.crf.viterbi_decode(real_logit, transitions)
        end = time.time()
        print("time: {:.4f}".format(end-start))
        # decode crf
        output = []
        word = ""
        for char_id, mark in zip(inputs, decoded_seq):
            char = self.vocab[char_id]
            if mark == 0:
                word = char
                output.append(word)
            if mark == 1:
                word = char
            if mark == 2:
                word += char
            if mark == 3:
                word += char
                output.append(word)

        print("\n".join(output))

if __name__ == "__main__":
    segmentor = Segmentor("data/char_pepole_vec.txt",
    "model/segment_model.pbtxt", "segment", 80)
    #segmentor.test_accuracy("model/test.txt")
    while True:
        sentence = input(">")
        segmentor.segment(sentence)

