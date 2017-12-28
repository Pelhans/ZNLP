#!/bin/bash

#dependency
mv ../../data/dependency/*.pkl ../dependency/data/pkl/

#cws
mv ../../data/cws_blstm/ckpt/ ../lexical_analysis/cws_blstm/
rm ../lexical_analysis/cws_blstm/ckpt/__init__
mv ../../data/cws_blstm/*.pkl ../lexical_analysis/cws_blstm/data/

#ner
mv ../../data/ner_blstm/ckpt/ ../lexical_analysis/ner_blstm/
rm ../lexical_analysis/ner_blstm/ckpt/__init__
mv ../../data/ner_blstm/*.pkl ../lexical_analysis/ner_blstm/data/

#pos
mv ../../data/pos_blstm/ckpt/ ../lexical_analysis/pos_blstm/
rm ../lexical_analysis/ner_blstm/ckpt/__init__
mv ../../data/pos_blstm/*.pkl ../lexical_analysis/pos_blstm/data/
