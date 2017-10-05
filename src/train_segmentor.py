import tensorflow as tf
import numpy as np
import word2vec as w2v
from bi_lstm_crf import Model as BiLstmModel
from id_cnn_crf import Model as IdCnnModel
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
tf.flags.DEFINE_string("debug_data_path",
        "data/debug.txt", "for debug display")
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

    cnn_layers = [
             {'rate' : 1},
             {'rate' : 1},
             {'rate' : 2}
            ]

    if FLAGS.using_lstm:
        model = BiLstmModel(vocab_size, embedding_size,
            FLAGS.hidden_size,
            FLAGS.lstm_layers,
            FLAGS.keep_rate,
            FLAGS.class_num,
            FLAGS.learning_rate,
            FLAGS.gradients_clip,
            FLAGS.max_seq_length
            )
    else:
        model = IdCnnModel(vocab_size, embedding_size,
                    FLAGS.filter_num,
                    FLAGS.filter_height,
                    3,
                    cnn_layers,
                    FLAGS.keep_rate,
                    FLAGS.class_num,
                    FLAGS.learning_rate,
                    FLAGS.max_seq_length)

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
            assert len(ints) == FLAGS.max_seq_length * 2
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
            assert len(ints) == FLAGS.max_seq_length * 2
            validate_inputs.append(ints)
        print("reading {} lines validate data".format(len(validate_inputs)))
        validate_inputs_np = np.array(validate_inputs, dtype="int32")

    # training

    lexicon = {w:i for i, w in enumerate(w2v_model.vocab)}
    model.train(train_inputs_np, validate_inputs_np, FLAGS.max_train_steps,
            FLAGS.batch_size, w2v_model.vectors, FLAGS.debug_data_path, lexicon)

if __name__ == "__main__":
    tf.app.run()
