# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import word2vec as w2v
import time
import user_dict_seg as ud

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("segment_model_path", "", "the segmentor model path")
tf.flags.DEFINE_string("char_vocab_path", "", "the char vocab path")
tf.flags.DEFINE_string("kcws_char_vocab_path", "", "the char vocab of kcws path")
tf.flags.DEFINE_string("word_vocab_path", "", "the word vocab path")
tf.flags.DEFINE_string("ner_model_path", "", "the ner model path")
tf.flags.DEFINE_string("user_dict_path", "", "the user dict for segment")

def get_char_vob(path):
    char_vob = {}
    with open(path, "r") as f:
        for line in f:
           char, char_id = line.split("\t")
           char_vob[char] = int(char_id)
    return char_vob

class Segmentor:
    """ Using kcws model to do segmentor, **be careful of the char vob"
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
        #model = w2v.load(lexicon_path)
        #self.lexicon = {w:i for i, w in enumerate(model.vocab)}
        #self.unk = self.lexicon['<UNK>']
        self.lexicon = get_char_vob(lexicon_path)
        self.unk = self.lexicon['<UNK>']

        # load model
        with tf.gfile.GFile(model_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, input_map=None, return_elements=None,
                name=model_prefix, op_dict=None, producer_op_list=None)

            self.x_inputs = graph.get_tensor_by_name("%s/input_placeholder:0" % model_prefix)
            self.logits = graph.get_tensor_by_name("%s/Reshape_7:0" % model_prefix)
            self.transition_params = graph.get_tensor_by_name("%s/transitions:0" %
                model_prefix)
            self.x_inputs_length = tf.reduce_sum(tf.sign(self.x_inputs), axis=1)

            self.sess = tf.Session(graph=graph)

        #user dict
        self.trie_tree = ud.TrieTree()
        self.trie_tree.read_from_file(user_dict_path)

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

        real_lengths = np.sum(np.sign(x_inputs_val), axis=1)

        start = time.time()
        logits_val, transitions = self.sess.run(
                [self.logits, self.transition_params],
                {self.x_inputs: x_inputs_val})

        # TODO using user defined dict
        decoded_results = []
        for logit, real_length, sentence in zip(logits_val, real_lengths,
                split_sentences):
            real_logit = logit[:real_length]
            if self.trie_tree.num_node > 1:
                scorer = ud.UserScore(sentence, self.trie_tree)
                user_score = scorer.get_score()
                real_logit += user_score
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

        return results

class Ner:
    """ name entity recognition
    """
    def __init__(self, char_lexicon_path, word_lexicon_path, model_path,
            model_prefix, max_sentence_length, max_word_length):
        self.max_sentence_length = max_sentence_length
        self.max_word_length = max_word_length
        self.model_prefix = model_prefix

        # load char lexicon
        char_model = w2v.load(char_lexicon_path)
        self.char_lexicon = {w:i for i, w in enumerate(char_model.vocab)}
        self.char_unk = self.char_lexicon['<UNK>']

        # load word lexicon
        word_model = w2v.load(word_lexicon_path)
        self.word_lexicon = {w:i for i, w in enumerate(word_model.vocab)}
        self.word_unk = self.word_lexicon['<UNK>']

        # load model
        with tf.gfile.GFile(model_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, input_map=None, return_elements=None,
                name=model_prefix, op_dict=None, producer_op_list=None)

            self.word_inputs = graph.get_tensor_by_name("%s/input_word:0" % model_prefix)
            self.char_inputs = graph.get_tensor_by_name("%s/input_char:0" % model_prefix)
            self.logits = graph.get_tensor_by_name("%s/logits_crf:0" % model_prefix)
            self.transition_params = graph.get_tensor_by_name("%s/transitions:0" %
                model_prefix)

            self.sess = tf.Session(graph=graph)

    def get_result(self, tokens):
        """ tagging the token list

        Args:
            tokens: <word> <word>
        """
        print(tokens)
        word_inputs = np.zeros([1, self.max_sentence_length], dtype="int32")
        char_inputs = np.zeros([1, self.max_sentence_length *
            self.max_word_length], dtype="int32")

        for i in range(len(tokens)):
            word = tokens[i]
            word_id = self.word_lexicon[word] if word in self.word_lexicon else self.word_unk
            word_inputs[0, i] = word_id
            for j in range(len(word)):
                char = word[j]
                char_id = self.char_lexicon[char] if char in self.char_lexicon else self.char_unk
                char_inputs[0, i * self.max_word_length + j] = char_id

        real_lengths = np.sum(np.sign(word_inputs), axis=1)
        start = time.time()
        logits_val, transitions = self.sess.run(
                [self.logits, self.transition_params],
                {self.word_inputs: word_inputs, self.char_inputs: char_inputs})

        logit = logits_val[0]
        real_length = real_lengths[0]
        real_logit = logit[:real_length]
        print(real_logit)
        decoded_seq, _ = tf.contrib.crf.viterbi_decode(real_logit, transitions)

        end = time.time()
        print("time: {:.4f}".format(end-start))

        for word, tag in zip(tokens, decoded_seq):
            print(word, tag)


def main(_):
    segmentor = Segmentor(FLAGS.user_dict_path, FLAGS.kcws_char_vocab_path,
            FLAGS.segment_model_path, "segment", 80)
    ner = Ner(FLAGS.char_vocab_path, FLAGS.word_vocab_path,
            FLAGS.ner_model_path, "ner", 50, 5)
    while True:
            sentence = input(">")
            tokens = segmentor.segment(sentence)
            ner.get_result(tokens)

if __name__ == "__main__":
    tf.app.run()
