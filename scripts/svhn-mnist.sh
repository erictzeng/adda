#!/bin/bash

# abort entire script on error
set -e

# train base model on svhn
python tools/train.py svhn train lenet lenet_svhn \
       --iterations 10000 \
       --batch_size 128 \
       --display 10 \
       --lr 0.001 \
       --snapshot 5000 \
       --solver adam

# run adda svhn->mnist
python tools/train_adda.py svhn:train mnist:train lenet adda_lenet_svhn_mnist \
       --iterations 10000 \
       --batch_size 128 \
       --display 10 \
       --lr 0.0002 \
       --snapshot 5000 \
       --weights snapshot/lenet_svhn \
       --adversary_relu \
       --solver adam

# evaluate trained models
echo 'Source only baseline:'
python tools/eval_classification.py mnist train lenet snapshot/lenet_svhn

echo 'ADDA':
python tools/eval_classification.py mnist train lenet snapshot/adda_lenet_svhn_mnist
