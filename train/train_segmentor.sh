#! /bin/sh
source ~/.bash_profile

#python src/process_pepole_data.py data/2014 data/char_pepole.txt

#src/word2vec/word2vec -train data/char_pepole.txt -save-vocab data/char_pepole_vocab.txt -min-count 3

#python src/replace_unk.py data/char_pepole_vocab.txt data/char_pepole.txt data/char_pepole_unk.txt

#src/word2vec/word2vec -train data/char_pepole_unk.txt   \
#                      -output data/char_pepole_vec.txt  \
#                      -size 50 -sample 1e-4 -negative 5 -hs 1 -binary 0 -iter 5 -min-count 3

#python src/generate_training.py data/char_pepole_vec.txt data/2014 data/all_data.txt

#python src/split_data.py data/all_data.txt

python train/freeze_graph.py --input_graph model/graph.pbtxt --input_checkpoint model/best_model --output_node_names "transitions,logits_crf,test_input_x" --output_graph model/segment_model.pbtxt
