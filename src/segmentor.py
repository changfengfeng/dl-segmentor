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
    while True:
        sentence = input(">")
        segmentor.segment(sentence)

