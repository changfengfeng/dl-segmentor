# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import word2vec as w2v
from bi_lstm_crf_queue import Model as BiLstmModel
from id_cnn_crf_queue import Model as IdCnnModel

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer("hidden_size", 100, "")
tf.flags.DEFINE_integer("lstm_layers", 1, "")
tf.flags.DEFINE_float("keep_rate", 0.5, "")
tf.flags.DEFINE_float("class_num", 4, "")
tf.flags.DEFINE_float("learning_rate", 0.001, "")
tf.flags.DEFINE_float("gradients_clip", 5 , "")
tf.flags.DEFINE_integer("max_seq_length", 80, "")
tf.flags.DEFINE_string("train_data_path",
        "data/train.txt", "")
tf.flags.DEFINE_string("validate_data_path",
        "data/test.txt", "")
tf.flags.DEFINE_string("embedding_path",
        "data/char_pepole_vec.txt", "")
tf.flags.DEFINE_integer("max_train_steps", 150000, "")
tf.flags.DEFINE_integer("batch_size", 100, "")
tf.flags.DEFINE_bool("using_lstm", True, "")
tf.flags.DEFINE_integer("filter_height", 3, "")
tf.flags.DEFINE_integer("filter_num", 100, "")
tf.flags.DEFINE_string("log_dir", "log/segmentor", "")

def main(_):
    w2v_model = w2v.load(FLAGS.embedding_path)
    vocab_size, embedding_size = w2v_model.vectors.shape
    print("vocab {}, embedding_size {}".format(vocab_size,
        embedding_size))
    if FLAGS.using_lstm:
        model = BiLstmModel(
            w2v_model.vectors,
            FLAGS.hidden_size,
            FLAGS.lstm_layers,
            FLAGS.class_num,
            FLAGS.learning_rate,
            FLAGS.keep_rate,
            FLAGS.gradients_clip,
            FLAGS.max_seq_length,
            FLAGS.log_dir,
            FLAGS.max_train_steps
            )
    else:
        cnn_layers = [
             {'rate' : 1},
             {'rate' : 1},
             {'rate' : 2}
            ]
        model = IdCnnModel(
            w2v_model.vectors,
            embedding_size,
            FLAGS.filter_num,
            FLAGS.filter_height,
            3,
            cnn_layers,
            FLAGS.keep_rate,
            FLAGS.class_num,
            FLAGS.learning_rate,
            FLAGS.max_seq_length,
            FLAGS.log_dir,
            FLAGS.max_train_steps)

    model.train(FLAGS.batch_size, FLAGS.train_data_path,
            FLAGS.validate_data_path)

if __name__ == "__main__":
    tf.app.run()
