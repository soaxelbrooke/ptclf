#!/bin/bash

echo "Checking for CUDA and installing."
# Check for CUDA and try to install.
if ! dpkg-query -W cuda-8-0; then
    # The 16.04 installer works with 16.10.
    curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
    dpkg -i ./cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
    apt-get update
    apt-get install cuda-8-0 -y
fi

echo "Installing python3.6..."
add-apt-repository ppa:jonathonf/python-3.6 -y
apt-get update -y
apt-get install python3.6 python3.6-dev python-apt -y
ln -s `which python3.6` /bin/python
echo "Installing pip..."
wget https://bootstrap.pypa.io/get-pip.py
python3.6 get-pip.py
ln -s `which pip` /bin/pip

echo "Setting up ptclf.py..."
gsutil cp gs://axelbrooke-data/ptclf/requirements.txt ./
gsutil cp gs://axelbrooke-data/ptclf/ptclf.py ./
python3.6 -m pip install -r requirements.txt


export EMBED_DIM=200
export PYTHONIOENCODING=utf8

echo "Downloading data..."
gsutil cp gs://axelbrooke-data/sentiment-tmp/train.csv ./train.csv
gsutil cp gs://axelbrooke-data/sentiment-tmp/dev.csv ./dev.csv
gsutil cp gs://axelbrooke-data/glove/glove.twitter.27B.$(echo "$EMBED_DIM")d.txt ./glove.txt

export COMET_API_KEY=DQqhNiimkjP0gK6c8iGz9orzL
export COMET_PROJECT=ptclf_sentiment 

export EPOCH_SHELL_CALLBACK='gsutil cp model.* gs://axelbrooke-models/ptclf-sentiment/$(date +%Y-%m-%d)_$(cat model.toml | grep "id =" | cut -c 7-42)/'

echo "Starting ptclf training..."
python3.6 ptclf.py train -m model -i train.csv --validate_path dev.csv --cuda \
    --glove_path ./glove.txt \
    --context_dim 64 \
    --embed_dim $EMBED_DIM \
    --vocab_size 16384 \
    --epochs 10 \
    --rnn_layers 2 \
    --rnn lstm \
    --embed_dropout 0.3 \
    --context_dropout 0.3 \
    --learn_rnn_init \
    --verbose 2 \
    --learning_rate 0.01 \
    --msg_len 70 \
    --classes positive,neutral,negative \
    --preload_data \
    --batch_size 1024

shutdown now
