#!/usr/bin/env bash

./ptclf.py train -m test-model -i combined_train.csv --classes positive,neutral,negative \
    --limit 1000
./ptclf.py train -m test-model -i combined_train.csv --limit 1000 --continued
./ptclf.py train -m test-model -i combined_train.csv --limit 1000 --continued \
    --validate_path combined_dev.csv --cuda
./ptclf.py predict -m test-model -i ~/tmp/sentiment_in.csv --cuda > /dev/null

rm test-model.*
