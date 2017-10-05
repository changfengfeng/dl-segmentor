# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import word2vec as w2v
import time
import user_dict_seg as ud

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_bool("using_lstm", False, "using lstm or id cnn model")
tf.flags.DEFINE_integer("mode", 0, "0: test the test.txt, 1: segment the stdin, 2: segment the file")
tf.flags.DEFINE_string("segment_input_fn", "", "file to segment it content")
tf.flags.DEFINE_bool("debug", True, "wheather to print the debug message")

class Segmentor:
    """ Import lexicon and model, segment text
    """

    def __init__(self, user_dict_path, lexicon_path, model_path, model_prefix, max_seq_length):

        self.pair_marks = {"(": ")",
                "（" : "）",
                "["  : "]",
                "【" : "】",
                "《" : "》",
                '“'  : '”'}

        self.break_marks = set(["。", ",", "，", " ", "\t", "?", "？", "!",
            "！", ";", "；"])
        self.pair_marks_reverse = {v:k for k, v in self.pair_marks.items()}

        self.max_seq_length = max_seq_length
        self.lexicon_path = lexicon_path
        self.model_path = model_path
        self.model_prefix = model_prefix

        # load lexicon
        model = w2v.load(lexicon_path)
        self.lexicon = {w:i for i, w in enumerate(model.vocab)}
        self.unk = self.lexicon['<UNK>']

        # load model
        with tf.gfile.GFile(model_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        tf.import_graph_def(graph_def, input_map=None, return_elements=None,
            name=model_prefix, op_dict=None, producer_op_list=None)

        graph = tf.get_default_graph()
        self.x_inputs = graph.get_tensor_by_name("%s/test_input_x:0" % model_prefix)
        self.logits = graph.get_tensor_by_name("%s/logits_crf:0" % model_prefix)
        self.transition_params = graph.get_tensor_by_name("%s/transitions:0" %
                model_prefix)
        self.sess = tf.Session()

        #user dict
        self.trie_tree = ud.TrieTree()
        if len(user_dict_path) > 0:
            self.trie_tree.read_from_file(user_dict_path)

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

        logits_val, transitions, real_lengths = self.sess.run(
                [self.logits, self.transition_params,
                    test_real_length],
                {self.x_inputs: x_inputs_array})

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

    def segment(self, sentence):
        """segment sentence

        Args:
            sentence: the input string
        Returns:
            list of string
        """
        split_sentences = self.split_text(sentence)
        x_inputs_val = np.zeros([len(split_sentences), self.max_seq_length], dtype="int32")
        for i in range(len(split_sentences)):
            for j in range(len(split_sentences[i])):
                s = split_sentences[i][j]
                sid = self.lexicon[s] if s in self.lexicon else self.unk
                x_inputs_val[i,j] = sid
        if FLAGS.debug:
            print(split_sentences)
            print(x_inputs_val)

        real_lengths = np.sum(np.sign(x_inputs_val), axis=1)

        start = time.time()
        logits_val, transitions = self.sess.run(
                [self.logits, self.transition_params],
                {self.x_inputs: x_inputs_val})

        decoded_results = []
        for logit, real_length, sentence in zip(logits_val, real_lengths,
                split_sentences):
            real_logit = logit[:real_length]
            if FLAGS.debug:
                print(real_logit)
            if self.trie_tree.num_node > 1:
                scorer = ud.UserScore(sentence, self.trie_tree)
                user_score = scorer.get_score()
                if FLAGS.debug:
                    print(user_score)
                real_logit += user_score
            if FLAGS.debug:
                print(real_logit)
            decoded_seq, _ = tf.contrib.crf.viterbi_decode(real_logit, transitions)
            decoded_results.append(decoded_seq)
        end = time.time()
        print("time: {:.4f}".format(end-start))

        # decode crf
        results = []
        for sentence, tags in zip(split_sentences, decoded_results):
            assert len(sentence) == len(tags)
            word = ""
            for s, t in zip(sentence, tags):
                if t == 0:
                    results.append(s)
                if t == 1:
                    word = s
                if t == 2:
                    word += s
                if t == 3:
                    word += s
                    results.append(word)
                    word = ""

        for s in results:
            print("tok:", s)

def main(_):
    if FLAGS.using_lstm:
        model_fn = "model/segment_model_30000.pbtxt"
    else:
        model_fn = "model/segment_model_idcnn_30000.pbtxt"

    segmentor = Segmentor("",
            "model/vec.txt", model_fn, "segment", 80)
    if FLAGS.mode == 0:
        segmentor.test_accuracy("model/test.txt")
    elif FLAGS.mode == 1:
        while True:
            sentence = input(">")
            segmentor.segment(sentence)
    elif FLAGS.mode == 2:
        with open(FLAGS.segment_input_fn, "r") as f:
            for line in f:
                print(line.strip())
                segmentor.segment(line)
                print("\n")

if __name__ == "__main__":
    tf.app.run()
