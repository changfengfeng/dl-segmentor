import tensorflow as tf
import numpy as np
import word2vec as w2v
from bi_lstm_cnn_crf import Model as BiLstmCnnModel

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer("max_sequence_length", 50, "")
tf.flags.DEFINE_integer("max_word_length", 5, "")

tf.flags.DEFINE_string("char_embedding_path",
        "data/char_pepole_vec.txt", "")
tf.flags.DEFINE_string("word_embedding_path",
        "data/word_pepole_vec.txt", "")

tf.flags.DEFINE_integer("filter_height", 2, "")
tf.flags.DEFINE_integer("filter_num", 50, "")

tf.flags.DEFINE_integer("hidden_size", 100, "")
tf.flags.DEFINE_integer("lstm_layers", 1, "")
tf.flags.DEFINE_float("keep_rate", 0.5, "")
tf.flags.DEFINE_float("class_num", 74, "")
tf.flags.DEFINE_float("learning_rate", 0.001, "")
tf.flags.DEFINE_float("gradients_clip", 7 , "")

tf.flags.DEFINE_string("train_data_path",
        "data/train_pos.txt", "")
tf.flags.DEFINE_string("validate_data_path",
        "data/test_pos.txt", "")
tf.flags.DEFINE_integer("max_train_steps", 50000, "")
tf.flags.DEFINE_integer("batch_size", 64, "")

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
                FLAGS.max_sequence_length,
                FLAGS.max_word_length,
                w2v_model.vectors,
                c2v_model.vectors,
                FLAGS.filter_height,
                FLAGS.filter_num,
                FLAGS.hidden_size,
                FLAGS.lstm_layers,
                FLAGS.keep_rate,
                FLAGS.class_num,
                FLAGS.learning_rate,
                FLAGS.gradients_clip
            )

    # loading data
    with open(FLAGS.train_data_path, "r") as f:
        train_inputs = []
        while True:
            line = f.readline()
            line = line.strip()
            if line == None or len(line) == 0:
                break
            ints =  np.array(line.split(" "))
            ints = ints.astype(int)
            assert len(ints) == FLAGS.max_sequence_length * 2 + FLAGS.max_sequence_length * FLAGS.max_word_length;
            train_inputs.append(ints)
        print("reading {} lines training data".format(len(train_inputs)))
        train_inputs_np = np.array(train_inputs, dtype="int32")

    with open(FLAGS.validate_data_path, "r") as f:
        validate_inputs = []
        while True:
            line = f.readline()
            line = line.strip()
            if line == None or len(line) == 0:
                break
            ints =  np.array(line.split(" "))
            ints = ints.astype(int)
            assert len(ints) == FLAGS.max_sequence_length * 2 + FLAGS.max_sequence_length * FLAGS.max_word_length
            validate_inputs.append(ints)
        print("reading {} lines validate data".format(len(validate_inputs)))
        validate_inputs_np = np.array(validate_inputs, dtype="int32")

    # training
    model.train(train_inputs_np, validate_inputs_np, FLAGS.max_train_steps,
            FLAGS.batch_size)

if __name__ == "__main__":
    tf.app.run()
