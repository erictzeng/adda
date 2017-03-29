# Adversarial Discriminative Domain Adaptation

## Getting started

This code requires Python 3, and is implemented in Tensorflow.

Hopefully things should be fairly easy to run out of the box:

    pip install -r requirements.txt
    mkdir data snapshot
    export PYTHONPATH="$PWD:$PYTHONPATH"
    scripts/svhn-mnist.sh

The provided script does the following things:

- Train a base LeNet model on SVHN (downloading SVHN under `data/svhn` in the process)
- Use ADDA to adapt the SVHN model to MNIST (downloading MNIST under `data/mnist` in the process)
- Run an evaluation on MNIST using the source-only model (stored at `snapshot/lenet_svhn`)
- Run an evaluation on MNIST using the ADDA model (stored at `snapshot/adda_lenet_svhn_mnist`)

## Areas of interest

- Check `scripts/svhn-mnist.sh` for hyperparameters.
- The LeNet model definition is in `adda/models/lenet.py`.
- The model is annotated with data preprocessing info, which is used in the `preprocessing` function in `adda/models/model.py`.
- The main ADDA logic happens in `tools/train_adda.py`.
- The adversarial discriminator model definition is in `adda/adversary.py`.

