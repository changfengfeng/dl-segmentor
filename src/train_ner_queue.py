import tensorflow as tf
import numpy as np
import word2vec as w2v
from bi_lstm_cnn_crf_queue import Model as BiLstmCnnModel

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer("max_sentence_length", 50, "")
tf.flags.DEFINE_integer("max_word_length", 5, "")

tf.flags.DEFINE_string("char_embedding_path",
        "data/char_pepole_vec.txt", "")
tf.flags.DEFINE_string("word_embedding_path",
        "data/word_pepole_vec.txt", "")

tf.flags.DEFINE_integer("filter_height", 2, "")
tf.flags.DEFINE_integer("filter_num", 50, "")

tf.flags.DEFINE_integer("hidden_size", 100, "")
tf.flags.DEFINE_integer("lstm_layers", 1, "")
tf.flags.DEFINE_float("class_num", 13, "BMES for 3 type + other")
tf.flags.DEFINE_float("learning_rate", 0.01, "")
tf.flags.DEFINE_float("gradients_clip", 7 , "")

tf.flags.DEFINE_string("train_data_path",
        "data/train_ner.txt", "")
tf.flags.DEFINE_string("validate_data_path",
        "data/test_ner.txt", "")
tf.flags.DEFINE_integer("max_train_steps", 50000, "")
tf.flags.DEFINE_integer("batch_size", 64, "")
tf.flags.DEFINE_string("log_dir", "log/ner", "")

def main(_):

    w2v_model = w2v.load(FLAGS.word_embedding_path)
    c2v_model = w2v.load(FLAGS.char_embedding_path)
    word_vocab_size, word_embedding_size = w2v_model.vectors.shape
    char_vocab_size, char_embedding_size = c2v_model.vectors.shape

    print("word_vocab {}, word_embedding_size {}".format(word_vocab_size,
        word_embedding_size))
    print("char_vocab {}, char_embedding_size {}".format(char_vocab_size,
        char_embedding_size))

    model = BiLstmCnnModel(
                FLAGS.max_sentence_length,
                FLAGS.max_word_length,
                w2v_model.vectors,
                c2v_model.vectors,
                FLAGS.filter_height,
                FLAGS.filter_num,
                FLAGS.hidden_size,
                FLAGS.lstm_layers,
                FLAGS.class_num,
                FLAGS.learning_rate,
                FLAGS.gradients_clip,
                FLAGS.log_dir,
                FLAGS.max_train_steps
            )

    model.train(FLAGS.batch_size, FLAGS.train_data_path,
            FLAGS.validate_data_path)

if __name__ == "__main__":
    tf.app.run()
