#!/bin/bash
mkdir -p ../../data/cws_blstm/ckpt/
mkdir -p ../../data/ner_blstm/ckpt/
mkdir -p ../../data/pos_blstm/ckpt/
#dependency
mv ../dependency/data/pkl/* ../../data/dependency/

#cws
mv ../lexical_analysis/cws_blstm/ckpt/* ../../data/cws_blstm/ckpt/
touch ../lexical_analysis/cws_blstm/ckpt/__init__
mv ../lexical_analysis/cws_blstm/data/*.pkl ../../data/cws_blstm/

#ner
mv ../lexical_analysis/ner_blstm/ckpt/* ../../data/ner_blstm/ckpt/
touch ../lexical_analysis/ner_blstm/ckpt/__init__
mv ../lexical_analysis/ner_blstm/data/*.pkl ../../data/ner_blstm/

#pos
mv ../lexical_analysis/pos_blstm/ckpt/* ../../data/pos_blstm/ckpt/
touch ../lexical_analysis/pos_blstm/ckpt/__init__
mv ../lexical_analysis/pos_blstm/data/*.pkl ../../data/pos_blstm/
