#!/bin/bash
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
sh install_fairseq.sh
cd examples/language_model/
bash prepare-wikitext-103.sh
cd ../..
mkdir -p ~/dataset/data-bin/
TEXT=examples/language_model/wikitext-103
fairseq-preprocess     --only-source     --trainpref $TEXT/wiki.train.tokens     --validpref $TEXT/wiki.valid.tokens     --testpref $TEXT/wiki.test.tokens     --destdir ~/dataset/data-bin/wikitext-103     --workers 20