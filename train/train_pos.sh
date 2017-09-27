#! /bin/sh

source ~/.bash_profile

echo "_____prepare"
#python src/process_pepole_data_pos.py data/2014 data/word_pepole.txt

echo "_____vocab"
#src/word2vec/word2vec \
#    -train data/word_pepole.txt  \
#    -save-vocab data/word_pepole_vocab.txt  \
#    -min-count 5

echo "_____unk"
#python src/replace_unk.py data/word_pepole_vocab.txt data/word_pepole.txt data/word_pepole_unk.txt

echo "_____word2vec"
#src/word2vec/word2vec -train data/word_pepole_unk.txt -output data/word_pepole_vec.txt -size 150 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0  -cbow 0 -iter 3 -min-count 5 -hs 1

echo "_____create pos vocab"
#python src/stats_pos.py data/2014 data/pos_vocab.txt data/word_pos.txt

echo "_____generate training data"
#python src/generate_pos_train.py \
#    data/word_pepole_vec.txt \
#    data/char_pepole_vec.txt \
#    data/pos_vocab.txt \
#    data/2014 \
#    data/all_pos_data.txt

#sort -u data/all_pos_data.txt > data/all_pos_data.u
#mv data/all_pos_data.u data/all_pos_data.txt
head -n 270000 data/all_pos_data.txt > data/train_pos.txt
tail -n 5000 data/all_pos_data.txt > data/test_pos.txt
