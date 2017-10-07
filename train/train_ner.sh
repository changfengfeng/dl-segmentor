#! /bin/sh

source ~/.bash_profile

echo "_____prepare"
#python src/process_pepole_data_ner.py data/2014 data/ner_pepole.txt

echo "_____word vocab"
#src/word2vec/word2vec \
#    -train data/ner_pepole.txt \
#    -save-vocab data/ner_pepole_vocab.txt \
#    -min-count 5

echo "_____unk"
#python src/replace_unk.py data/ner_pepole_vocab.txt data/ner_pepole.txt data/ner_pepole_unk.txt

echo "_____word2vec"
#src/word2vec/word2vec -train data/ner_pepole_unk.txt -output data/ner_pepole_vec.txt -size 150 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0  -cbow 0 -iter 3 -min-count 5 -hs 1

echo "_____generate trainning data"
python src/generate_ner_train.py \
    data/ner_pepole_vec.txt \
    data/char_pepole_vec.txt \
    data/2014 \
    data/all_ner_data.txt

sort -u data/all_ner_data.txt > data/all_ner_data.u
mv data/all_ner_data.u data/all_ner_data.txt

head -n 178121 data/all_ner_data.txt > data/train_ner.txt
tail -n 2000 data/all_ner_data.txt > data/test_ner.txt
