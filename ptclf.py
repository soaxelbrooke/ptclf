#!/usr/bin/env python3.6

import os
from datetime import datetime

import pickle
from comet_ml import Experiment

COMET_API_KEY = os.environ.get('COMET_API_KEY')
COMET_PROJECT = os.environ.get('COMET_PROJECT')
COMET_LOG_CODE = os.environ.get('COMET_LOG_CODE', '').lower() == 'true'

from unittest.mock import MagicMock
import argparse
import toolz
import pandas
import hashlib
import logging

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from collections import deque
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm
tqdm.monitor_interval = 0  # https://github.com/tqdm/tqdm/issues/481


class WordRnn(nn.Module):
    def __init__(self, args):
        super(WordRnn, self).__init__()

        self.embed_size = args.embeddim
        self.context_size = args.contextdim
        self.rnn_layers = args.rnnlayers
        self.rnn_kind = args.rnn
        self.input_size = args.vocabsize
        self.cuda = args.cuda
        self.seqlen = args.maxlen
        self.num_classes = len(args.classes.split(','))
        self.bidirectional = args.bidirectional

        self.embedding = nn.Embedding(self.input_size, self.embed_size, padding_idx=0)
        self.embed_dropout = nn.Dropout(args.embeddropout)

        if self.rnn_kind == 'gru':
           self.rnn = nn.GRU(self.embed_size, self.context_size, self.rnn_layers,
                             bidirectional=args.bidirectional)
        elif self.rnn_kind == 'lstm':
           self.rnn = nn.LSTM(self.embed_size, self.context_size, self.rnn_layers,
                              bidirectional=args.bidirectional)
        else:
           raise RuntimeError('Got invalid rnn type: {}'.format(self.rnn_kind))

        self.rnn_dropout = nn.Dropout(args.rnndropout)
        self.dense = nn.Linear(self.context_size * (1 + int(self.bidirectional)), self.num_classes)
        
        if self.cuda:
            self.embedding.cuda()
            self.rnn.cuda()
            self.dense.cuda()

        self.init_weights()

    def init_weights(self):
        # Embedding weight init
        nn.init.normal(self.embedding.weight)

        # RNN weight init
        for layer_params in self.rnn._all_weights:
            for param in layer_params:
                if 'weight' in param:
                    nn.init.xavier_normal(getattr(self.rnn, param))

    def forward(self, input_tensor):
        # input tensor shape: (msg_len, batch_size,)
        # hidden state shape: (rnn_depth, batch_size, context_size)
        batch_size = input_tensor.size(1)
        hidden_state = self.init_hidden(batch_size)
        embedded = self.embed_dropout(self.embedding(input_tensor))
        # embedded shape: (msg_len, batch_size, embed_size)
        context, hidden = self.rnn(embedded, hidden_state)
        last_context = self.rnn_dropout(context)[-1]
        # context shape: (msg_len, batch_size, context_size)
        # hidden shape: (rnn_depth, batch_size, context_size)
        dense = self.dense(last_context)
        # dense shape: (batch_size, num_classes)
        class_probs = F.softmax(dense, dim=1)
        # class probs shape: (batch_size, num_classes)
        return class_probs

    def init_hidden(self, batch_size):
        if self.rnn_kind == 'gru':
            hidden = torch.zeros(self.rnn_layers * (1 + int(self.bidirectional)),
                                 batch_size, self.context_size)
            return Variable(hidden.cuda()) if self.cuda else Variable(hidden)
        elif self.rnn_kind == 'lstm':
            hidden1 = torch.zeros(self.rnn_layers * (1 + int(self.bidirectional)), batch_size, self.context_size)
            hidden2 = torch.zeros(self.rnn_layers * (1 + int(self.bidirectional)), batch_size, self.context_size)
            return (Variable(hidden1.cuda()), Variable(hidden2.cuda())) if self.cuda else \
                    (Variable(hidden1), Variable(hidden2))


def parse_args():
    def env(name, default, transform):
        return transform(os.environ.get(name, default))

    parser = argparse.ArgumentParser(description='ptclf - Pytorch Text Classifier')
    
    parser.add_argument('mode', type=str,
                        help='Action to take - one of {train, test, predict}.')
    parser.add_argument('-i', '--inputpath', type=str,
                        help='Input file for training, testing, or prediction.')
    parser.add_argument('--validatepath', type=str,
                        help='Path to validation dataset during model training.')
    parser.add_argument('-m', '--modelpath', type=str,
                        help='Path to model for training, testing, or prediction')
    parser.add_argument('--epochs', type=int, default=env('EPOCHS', 1, int),
                        help='Number of epochs before terminating training')
    parser.add_argument('-b', '--batchsize', type=int, default=env('BATCHSIZE', 16, int))
    parser.add_argument('--cuda', action='store_true',
                        default=env('CUDA', 'False', lambda s: s.lower() == 'true'))
    parser.add_argument('--verbose', type=int, default=env('VERBOSE', 1, int),
                        help='Verbosity of model. 0 = silent, 1 = progress bar, 2 = one line '
                             'per epoch.')

    parser.add_argument('--lr', type=float, default=env('LR', 0.005, float))
    parser.add_argument('--optim', type=str, default=env('OPTIM', 'adam', str),
                        help='One of {sgd, adam}')
    parser.add_argument('--loss', type=str, default=env('LOSS', 'CrossEntropy', str))
    parser.add_argument('--embeddropout', type=float, default=env('EMBEDDROPOUT', 0.1, float),
                        help='Dropout used for embedding layer')
    parser.add_argument('--rnndropout', type=float, default=env('RNNDROPOUT', 0.1, float),
                        help='Dropout used for RNN output')

    parser.add_argument('--rnn', type=str, default='gru',
                        help='Type of RNN used - one of {gru, lstm}')
    parser.add_argument('--rnnlayers', type=int, default=1,
                        help='Number of RNN layers to stack')
    parser.add_argument('--bidirectional', action='store_true',
                        default=env('BIDIRECTIONAL', 'False', lambda s: s.lower() == 'true'),
                        help='If set, RNN is bidirectional')
    
    parser.add_argument('--charrnn', action='store_true', default=False,
                        help='Use a character RNN instead of word RNN')
    parser.add_argument('-v', '--vocabsize', type=int, default=env('VOCABSIZE', 2048, int),
                        help='Vocab size')
    parser.add_argument('--maxlen', type=int, default=env('MAXLEN', 40, int),
                        help='Maximum length of text in tokens')
    parser.add_argument('-c', '--contextdim', type=int, default=env('CONTEXTDIM', 64, int),
                        help='Dimension of the RNN context vector')
    parser.add_argument('-e', '--embeddim', type=int, default=env('EMBEDDIM', 100, int),
                        help='Dimension of the embedding (only used for word-RNN)')
    parser.add_argument('--tokenre', type=str, default=env('TOKENRE', r'\w+|\$[\d\.]+|\S+', str),
                        help='Regexp pattern to tokenize with')

    parser.add_argument('--classes', type=str,
                        help='Comma separated list of classes to predict')
    parser.add_argument('--classweights', type=str, default=env('CLASSWEIGHTS', None, lambda s: s),
                        help='Comma separated list of class weights')

    parser.add_argument('--continued', action='store_true',
                        help='Continue training')

    return parser.parse_args()


def args_to_comet_hparams(args):
    """ Turns command line args into a dict of hyperparams for reporting to comet """
    return {
        'input_path': args.inputpath,
        'validate_path': args.validatepath,
        'epochs': args.epochs,
        'batch_size': args.batchsize,
        'context_dim': args.contextdim,
        'embed_dim': args.embeddim,
        'token_regexp': args.tokenre,
        'classes': args.classes.split(','),
        'class_weights': [float(w) for w in args.classweights.split(',')] if args.classweights else None,
        'cuda': args.cuda,
        'learning_rate': args.lr,
        'optimizer': args.optim,
        'loss_function': args.loss,
        'rnn_type': args.rnn,
        'rnn_layers': args.rnnlayers,
        'char_rnn': args.charrnn,
        'vocab_size': args.vocabsize,
        'max_len': args.maxlen,
        'embed_dropout': args.embeddropout,
        'rnn_dropout': args.rnndropout,
        'bidirectional': args.bidirectional,
    }


@toolz.memoize
def get_tokenizer(tokenre):
    return RegexpTokenizer(tokenre)


def transform_texts(args, texts):
    """ Transforms texts based on provided args """
    if args.charrnn:
        raise NotImplementedError
    else:
        tensor = torch.zeros(args.maxlen, len(texts)).type(torch.LongTensor)
        tokenizer = get_tokenizer(args.tokenre)
        for row_idx, text in enumerate(texts):
            for col_idx, token in enumerate(toolz.take(args.maxlen, tokenizer.tokenize(text))):
                tensor[col_idx, row_idx] = \
                    (int(hashlib.md5(token.encode()).hexdigest(), 16) % (args.vocabsize - 1)) + 1

    return Variable(tensor.cuda()) if args.cuda else Variable(tensor)


def transform_classes(args, classes):
    """ Transforms classes based on provided args """
    class_dict = {cls: idx for idx, cls in enumerate(args.classes.split(','))}
    torch_classes = torch.LongTensor([class_dict[cls] for cls in classes])
    return Variable(torch_classes.cuda()) if args.cuda else Variable(torch_classes)


def train_batch_iter(args):
    yield from batch_iter_from_path(args, args.inputpath)


def dev_batch_iter(args):
    yield from batch_iter_from_path(args, args.validatepath)


def batch_iter_from_path(args, path):
    """ Loads, transforms, and yields batches for training/testing/prediction """
    chunk_iter = iter(pandas.read_csv(path, chunksize=args.batchsize, header=None))
    while True:
        try:
            super_chunk_size = 16
            super_chunk = pandas.concat([next(chunk_iter) for _ in range(super_chunk_size)]) \
                .sample(frac=1)

            for i in range(super_chunk_size):
                chunk = super_chunk[i*args.batchsize:(i+1)*args.batchsize]
                real_chunk = chunk.dropna(axis=0)
                yield transform_texts(args, real_chunk.loc[:, 0].values), \
                        transform_classes(args, real_chunk.loc[:, 1].values)
        except pandas.errors.ParserError:
            pass


def predict_batch_iter(args):
    chunk_iter = iter(pandas.read_csv(args.inputpath, chunksize=args.batchsize, header=None))
    while True:
        try:
            chunk = next(chunk_iter)
            real_chunk = chunk.dropna(axis=0)
            yield transform_texts(args, real_chunk.loc[:, 0].values)
        except pandas.errors.ParserError:
            pass


def build_model(args):
    """ Builds model based on arguments provided """
    if args.charrnn:
        raise NotImplementedError
    else:
        return WordRnn(args)


def infer_class_weights(args):
    """ Infers class weights from provided training data """
    logging.info('No class weights provided, inferring...')
    classes = args.classes.split(',')
    counts = pandas.Series({idx: 0 for idx in range(len(classes))})
    for batch_x, batch_y in train_batch_iter(args):
        counts += pandas.Series(batch_y.cpu().data).value_counts()

    weights = torch.FloatTensor([sum(counts) / counts[c] for c in range(len(classes))])
    logging.info('Inferred class weights: {}'.format(weights))
    return weights


def train(args):
    """ Trains RNN based on provided arguments """
    if args.continued:
        with open(args.modelpath + '.meta', 'rb') as infile:
            args = pickle.load(infile)
        model = WordRnn(args)
        model.load_state_dict(torch.load(args.modelpath + '.bin'))
        logging.info('Model loaded from {}, continuing training'.format(args.modelpath))
    else:
        model = build_model(args)

    class_weights = torch.FloatTensor([float(w) for w in args.classweights.split(',')]) if args.classweights \
        else infer_class_weights(args)
    if args.cuda:
        class_weights = class_weights.cuda()

    if args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    else:
        raise RuntimeError('Invalid optim value provided: {}'.format(args.optim))

    if args.loss == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    elif args.loss == 'NLL':
        criterion = nn.NLLLoss(weight=class_weights)
    else:
        raise RuntimeError('Invalid loss value provided: {}'.format(args.loss))

    if COMET_API_KEY:
        assert COMET_PROJECT is not None, 'You must specify a comet project to use if providing' \
                                          ' COMET_API_KEY environment variable.'
        comet_experiment = Experiment(api_key=COMET_API_KEY, project_name=COMET_PROJECT,
                                      log_code=COMET_LOG_CODE)
        comet_experiment.log_multiple_params(args_to_comet_hparams(args))
        comet_experiment.log_dataset_hash(open(args.inputpath).read())
    else:
        comet_experiment = MagicMock()

    try:
        for epoch in range(args.epochs):
            train_epoch(args, model, criterion, optimizer, epoch, comet_experiment)
            score_model(args, model, criterion, epoch, comet_experiment)
            torch.save(model.state_dict(), args.modelpath + '.bin')
            with open(args.modelpath + '.meta', 'wb') as metafile:
                pickle.dump(args, metafile)
            logging.info('Model saved at {}'.format(args.modelpath))
    except KeyboardInterrupt:
        pass

    try:
        while True:
            next_input = input('\n> ')
            print(predict_batch(model, transform_texts(args, [next_input])))
    except KeyboardInterrupt:
        print("\n\nBye!")


def train_epoch(args, model, criterion, optimizer, epoch, comet_experiment):
    """ Trains a single epoch """
    comet_experiment.log_current_epoch(epoch)
    loss_queue = deque(maxlen=100)
    all_losses = []
    progress = tqdm(desc='Epoch {}'.format(epoch)) if args.verbose == 1 else MagicMock()
    correct = 0
    seen = 0
    epoch_period = 1
    started = datetime.now()

    for step, (batch_x, batch_y) in enumerate(train_batch_iter(args)):
        output, loss = train_batch(model, criterion, optimizer, batch_x, batch_y)
        loss_queue.append(loss)
        all_losses.append(loss)
        rolling_loss = sum(loss_queue) / len(loss_queue)
        seen += batch_x.size(1)
        correct += float(sum((output.max(1)[1] == batch_y).data.cpu().numpy()))
        accuracy = correct / seen

        progress.set_postfix(loss=rolling_loss, acc=accuracy)
        progress.update(batch_x.size(1))

        comet_experiment.log_step(step)
        comet_experiment.log_loss(rolling_loss)
        comet_experiment.log_accuracy(accuracy)
        epoch_period = (datetime.now() - started).total_seconds()
        comet_experiment.log_metric('train_items_per_second', seen / epoch_period)

    progress.clear()
    if args.verbose > 0:
        logging.info('Epoch: {}\tTrain accuracy: {:.3f}\tTrain loss: {:.3f}\tTrain rate: {:.3f}'.format(
            epoch, correct/seen, sum(all_losses) / len(all_losses), seen / epoch_period))


def train_batch(model, criterion, optimizer, x, y):
    """ Handles a single batch """
    model.zero_grad()
    output = model(x)

    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    return output, loss.data[0]


def score_model(args, model, criterion, epoch, comet_experiment):
    """ Score model and update comet """
    losses = []
    correct = 0
    seen = 0
    for batch_x, batch_y in dev_batch_iter(args):
        model.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        losses.append(loss.data[0])
        seen += batch_x.size(1)
        correct += float(sum((output.max(1)[1] == batch_y).data.cpu().numpy()))

    comet_experiment.log_metric('dev_loss', sum(losses) / len(losses))
    comet_experiment.log_metric('dev_acc', correct / seen)
    if args.verbose > 0:
        logging.info('Epoch: {}\t  Dev accuracy: {:.3f}\t  Dev loss: {:.3f}'.format(
            epoch, correct/seen, sum(losses) / len(losses)))


def predict_batch(model, batch):
    model.zero_grad()
    output = model(batch)
    return output


def predict(args):
    with open(args.modelpath + '.meta', 'rb') as infile:
        model_args = pickle.load(infile)
        # model_args.cuda = False
    model = WordRnn(model_args)
    model.load_state_dict(torch.load(args.modelpath + '.bin'))

    classes = model_args.classes.split(',')

    for batch_x in predict_batch_iter(args):
        output = model(batch_x.cuda()).data
        if model_args.cuda:
            output = output.cpu()
        print('\n'.join(map(lambda idx: classes[idx], output.max(1)[1].numpy())))


def main():
    args = parse_args()

    logging.basicConfig(format='%(levelname)s:%(asctime)s.%(msecs)03d [%(threadName)s] - %(message)s',
                        datefmt='%Y-%m-%d,%H:%M:%S',
                        filename=os.environ.get('LOG_PATH'),
                        level=getattr(logging, os.environ.get('LOG_LEVEL', 'INFO')))

    if args.mode == 'train':
        train(args)
    elif args.mode == 'predict':
        predict(args)


if __name__ == '__main__':
    main()

