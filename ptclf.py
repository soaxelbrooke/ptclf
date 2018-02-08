#!/usr/bin/env python3.6
import csv
import os
from datetime import datetime
from uuid import uuid4

import numpy

COMET_API_KEY = os.environ.get('COMET_API_KEY')
COMET_PROJECT = os.environ.get('COMET_PROJECT')
COMET_LOG_CODE = os.environ.get('COMET_LOG_CODE', '').lower() == 'true'

if COMET_API_KEY:
    from comet_ml import Experiment

from unittest.mock import MagicMock
import argparse
import toolz
import pandas
import mmh3
import logging

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.backends import cudnn
cudnn.benchmark = True

import toml

from collections import deque, Counter
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm
tqdm.monitor_interval = 0  # https://github.com/tqdm/tqdm/issues/481


class WordRnn(nn.Module):
    def __init__(self, settings):
        """ Build model from settings
        :param Settings settings: Settings that describe the model to build
        """
        super(WordRnn, self).__init__()

        logging.debug('Creating model with the following settings:\n' + settings.to_toml())
        self.embed_size = settings.embed_dim
        self.context_size = settings.context_dim
        self.rnn_layers = settings.rnn_layers
        self.rnn_kind = settings.rnn
        self.input_size = settings.vocab_size
        self.cuda = settings.cuda
        self.gradient_clip = settings.gradient_clip
        self.seqlen = settings.msg_len
        self.num_classes = len(settings.classes)
        self.bidirectional = settings.bidirectional

        self.embedding = nn.Embedding(self.input_size, self.embed_size, padding_idx=0)
        if settings.glove_path:
            self.embedding.weight.data.copy_(load_glove(settings.glove_path, settings.vocab_size,
                                                        self.embed_size))

        self.embed_dropout = nn.Dropout(settings.embed_dropout)

        self.bidir_factor = 1 + int(self.bidirectional)
        rnn_hidden_shape = (settings.rnn_layers * self.bidir_factor, 1, self.context_size)

        if self.rnn_kind == 'gru':
            if settings.learn_rnn_init:
                self.rnn_init_h = nn.Parameter(
                    torch.randn(*rnn_hidden_shape).type(torch.FloatTensor), requires_grad=True)
            else:
                self.rnn_init_h = Variable(torch.zeros(
                    self.rnn_layers * (1 + int(self.bidirectional)), 1, self.context_size))
            self.rnn = nn.GRU(self.embed_size, self.context_size, self.rnn_layers,
                              bidirectional=self.bidirectional, dropout=settings.context_dropout)
        elif self.rnn_kind == 'lstm':
            if settings.learn_rnn_init:
                self.rnn_init_h1 = nn.Parameter(
                    torch.randn(*rnn_hidden_shape).type(torch.FloatTensor),requires_grad=True)
                self.rnn_init_h2 = nn.Parameter(
                    torch.randn(*rnn_hidden_shape).type(torch.FloatTensor),requires_grad=True)
            else:
                self.rnn_init_h1 = Variable(torch.zeros(
                    self.rnn_layers * (1 + int(self.bidirectional)), 1, self.context_size))
                self.rnn_init_h2 = Variable(torch.zeros(
                    self.rnn_layers * (1 + int(self.bidirectional)), 1, self.context_size))
            self.rnn = nn.LSTM(self.embed_size, self.context_size, self.rnn_layers,
                               bidirectional=self.bidirectional, dropout=settings.context_dropout)
        else:
            raise RuntimeError('Got invalid rnn type: {}'.format(self.rnn_kind))

        self.rnn_dropout = nn.Dropout(settings.context_dropout)
        self.dense = nn.Linear(self.context_size * self.bidir_factor, self.num_classes)

        if self.cuda:
            self.embedding.cuda()
            self.rnn.cuda()
            self.dense.cuda()

        self.init_weights()

    @classmethod
    def load(cls, settings):
        """ Load model from provided path
        :param Setting settings:
        :rtype: WordRnn
        """
        model = cls(settings)
        model.load_state_dict(torch.load(settings.model_path + '.bin'))
        return model

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
            init = self.rnn_init_h.repeat(1, batch_size, 1)
            return init.cuda() if self.cuda else init
        elif self.rnn_kind == 'lstm':
            init1 = self.rnn_init_h1.repeat(1, batch_size, 1)
            init2 = self.rnn_init_h2.repeat(1, batch_size, 1)
            return init1.cuda() if self.cuda else init1, init2.cuda() if self.cuda else init2


def load_glove(path, vocab_size, embed_dim):
    """ Load glove embeddings specified
    :param str path: path to load glove embeddings from
    :rtype: torch.FloatTensor
    """
    logging.info('Loading GLOVE embeddings from {}'.format(path))
    remaining = set(range(1, vocab_size))
    weights = numpy.zeros((vocab_size, embed_dim), dtype=float)
    with open(path) as infile:
        for line in infile:
            splits = line.split(' ')
            idx = word_to_idx(vocab_size, splits[0])
            if idx in remaining:
                weights[idx, :] = numpy.array(splits[1:], dtype=float)
                remaining.remove(idx)
            if len(remaining) == 0:
                break
    return torch.from_numpy(weights)


def load_settings_and_model(path, args=None):
    """ Load settings and model from the specified model path
    :param str path: model path (without .bin or .toml)
    :param Namespace args: optional args
    :rtype: (Settings, WordRnn)
    """
    settings = Settings.load(path + '.toml')
    if args:
        settings.add_args(args)
    model = WordRnn.load(settings)
    model.eval()
    return settings, model


def env(name, transform):
    var = os.environ.get(name)
    if var is not None:
        return transform(var)


def env_flag(name):
    return env(name, lambda s: s.lower() == 'true')


class Settings:
    model_param_names = {'id', 'created_at', 'rnn', 'rnn_layers', 'char_rnn', 'bidirectional',
                         'classes', 'vocab_size', 'msg_len', 'context_dim', 'embed_dim',
                         'learn_rnn_init'}
    default_names = {'batch_size', 'epochs', 'cuda', 'learning_rate', 'optimizer', 'loss_fn',
                     'embed_dropout', 'context_dropout', 'token_regex', 'class_weights',
                     'gradient_clip'}
    transient_names = {'input_path', 'validate_path', 'verbose', 'limit', 'glove_path',
                       'model_path', 'preload_data'}
    comet_hparam_names = {'rnn', 'rnn_layers', 'char_rnn', 'bidirectional', 'classes', 'vocab_size',
                          'msg_len', 'context_dim', 'embed_dim', 'batch_size', 'epochs', 'cuda',
                          'learning_rate', 'optimizer', 'loss_fn', 'embed_dropout',
                          'context_dropout', 'token_regex', 'class_weights'}

    # Default values for if no setting is provided for given parameter
    model_param_defaults = {
        'rnn': 'gru', 'rnn_layers': 1, 'bidirectional': True, 'char_rnn': False, 'vocab_size': 1024,
        'msg_len': 40, 'context_dim': 32, 'embed_dim': 50, 'learn_rnn_init': False,
    }

    default_defaults = {
        'epochs': 1, 'batch_size': 16, 'learning_rate': 0.005, 'optimizer': 'adam',
        'loss_fn': 'CrossEntropy', 'embed_dropout': 0.1, 'context_dropout': 0.1,
        'token_regex': r'\w+|\$[\d\.]+|\S+', 'gradient_clip': None,
    }

    transient_defaults = {'verbose': 1, 'glove_path': None, 'preload_data': False}

    def __init__(self, model_settings, defaults, transients):
        self.model_settings = model_settings
        self.defaults = defaults
        self.transients = transients
        self.try_defaults()

    def __getattr__(self, key):
        if key in self.model_param_names or key in self.default_names or key in self.transients:
            return self.get(key)
        else:
            raise AttributeError("'Settings' has no an attribute '{}'.".format(key))

    def get(self, key, default=None):
        if key in self.model_param_names:
            return self.model_settings.get(key, default)
        elif key in self.default_names:
            return self.defaults.get(key, default)
        elif key in self.transient_names:
            return self.transients.get(key, default)
        else:
            return default

    def try_defaults(self):
        """ Try to add defaults for missing settings """
        for name, default in self.model_param_defaults.items():
            if self.model_settings.get(name) is None:
                self.model_settings[name] = default
        for name, default in self.default_defaults.items():
            if self.defaults.get(name) is None:
                self.defaults[name] = default
        for name, default in self.transient_defaults.items():
            if self.transients.get(name) is None:
                self.transients[name] = default

    @classmethod
    def load(cls, path):
        """ Read settings from specified path
        :param str path: path to settings TOML file
        :rtype: Settings
        """
        with open(path) as infile:
            settings_dict = toml.load(infile)
        settings = Settings(settings_dict['model'], settings_dict.get('defaults', {}), {})
        settings.transients['model_path'] = path[:-5]
        return settings

    def save(self, path):
        """ Save TOML settings to provided path.
        :param str path: Path to save settings to.
        :rtype: NoneType
        """
        with open(path, 'w') as outfile:
            toml.dump({'model': self.model_settings, 'defaults': self.defaults}, outfile)

    def to_toml(self):
        return toml.dumps({'model': self.model_settings, 'defaults': self.defaults})

    @classmethod
    def parse_args(cls, args):
        """ Parses args into dicts for Settings build
        :param Any args: args from arg parser
        :rtype: (dict, dict, dict)
        """
        model_settings = {'id': str(uuid4()), 'created_at': datetime.utcnow()}
        defaults = {}
        transients = {}

        for name in cls.model_param_names:
            if name == 'classes' and args.classes:
                model_settings[name] = args.classes.split(',')
            elif name != 'id' and name != 'created_at':
                if getattr(args, name, None) is not None:
                    model_settings[name] = getattr(args, name)

        for name in cls.default_names:
            if name == 'class_weights' and args.class_weights:
                defaults[name] = [float(w) for w in args.class_weights.split(',')]
            elif getattr(args, name, None) is not None:
                defaults[name] = getattr(args, name)

        for name in cls.transient_names:
            if getattr(args, name, None) is not None:
                transients[name] = getattr(args, name)

        return model_settings, defaults, transients

    @classmethod
    def from_args(cls, args):
        """ Create new settings from command line args and env vars.
        :param Any args: args from `parse_args`
        :rtype: Settings
        """
        model_settings, defaults, transients = cls.parse_args(args)

        try:
            _settings = cls.load(args.model_path + '.toml')
            model_settings = toolz.merge(_settings.model_settings, model_settings)
            defaults = toolz.merge(_settings.defaults, defaults)
        except FileNotFoundError:
            pass

        settings = Settings(model_settings, defaults, transients)
        settings.try_defaults()
        return settings

    def add_args(self, args):
        """ Add settings command line args and env vars
        :param Any args: Args to override those loaded from file
        :rtype: NoneType
        """
        for name in self.default_names:
            if name == 'class_weights' and args.class_weights:
                self.defaults[name] = [float(w) for w in args.class_weights.split(',')]
            elif getattr(args, name, None):
                self.defaults[name] = getattr(args, name)
        for name in self.transient_names:
            if getattr(args, name, None):
                self.transients[name] = getattr(args, name)

    def to_comet_hparams(self):
        """ Turns settings into dict suitable for reporting to comet_ml
        :rtype: dict
        """
        return {name: self.get(name) for name in self.comet_hparam_names}


def parse_args():
    """ Parses command line args and env vars and adds them to current settings.
    :rtype: Any
    """
    parser = argparse.ArgumentParser(description='ptclf - Pytorch Text Classifier')

    parser.add_argument('mode', type=str,
                        help='Action to take - one of {train, test, predict}.')
    parser.add_argument('-i', '--input_path', type=str,
                        help='Input file for training, testing, or prediction.')
    parser.add_argument('--validate_path', type=str,
                        help='Path to validation dataset during model training.')
    parser.add_argument('-m', '--model_path', type=str,
                        help='Path to model for training, testing, or prediction')
    parser.add_argument('--epochs', type=int, default=env('EPOCHS', int),
                        help='Number of epochs before terminating training')
    parser.add_argument('-b', '--batch_size', type=int, default=env('BATCH_SIZE', int))
    parser.add_argument('--cuda', action='store_true', default=env_flag('CUDA'))
    parser.add_argument('--verbose', type=int, default=env('VERBOSE', int),
                        help='Verbosity of model. 0 = silent, 1 = progress bar, 2 = one line '
                             'per epoch.')

    parser.add_argument('--learning_rate', type=float, default=env('LEARNING_RATE', float))
    parser.add_argument('--gradient_clip', type=float, default=env('GRADIENT_CLIP', float))
    parser.add_argument('--optimizer', type=str, default=env('OPTIMIZER', str),
                        help='One of {sgd, adam}')
    parser.add_argument('--loss_fn', type=str, default=env('LOSS_FN', str)) # TODO add help details for other loss functions
    parser.add_argument('--embed_dropout', type=float, default=env('EMBED_DROPOUT', float),
                        help='Dropout used for embedding layer')
    parser.add_argument('--context_dropout', type=float, default=env('CONTEXT_DROPOUT', float),
                        help='Dropout used for RNN output')

    parser.add_argument('--rnn', type=str,
                        help='Type of RNN used - one of {gru, lstm}')
    parser.add_argument('--rnn_layers', type=int, default=env('RNN_LAYERS', int),
                        help='Number of RNN layers to stack')
    parser.add_argument('--bidirectional', action='store_true', default=env_flag('BIDIRECTIONAL'),
                        help='If set, RNN is bidirectional')
    parser.add_argument('--learn_rnn_init', action='store_true', default=env_flag('LEARN_RNN_INIT'),
                        help='Learn RNN initial state (default inits to tensor of 0s)')

    parser.add_argument('--char_rnn', action='store_true',
                        help='Use a character RNN instead of word RNN')
    parser.add_argument('-v', '--vocab_size', type=int, default=env('VOCAB_SIZE', int),
                        help='Vocab size')
    parser.add_argument('--msg_len', type=int, default=env('MSG_LEN', int),
                        help='Maximum length of text in tokens')
    parser.add_argument('-c', '--context_dim', type=int, default=env('CONTEXT_DIM', int),
                        help='Dimension of the RNN context vector')
    parser.add_argument('-e', '--embed_dim', type=int, default=env('EMBED_DIM', int),
                        help='Dimension of the embedding (only used for word-RNN)')
    parser.add_argument('--token_regex', type=str,
                        default=env('TOKEN_REGEX', str),
                        help='Regexp pattern to tokenize with')

    parser.add_argument('--classes', type=str,
                        help='Comma separated list of classes to predict')
    parser.add_argument('--class_weights', type=str,
                        default=env('CLASS_WEIGHTS', lambda s: s),
                        help='Comma separated list of class weights')

    parser.add_argument('--continued', action='store_true', help='Continue training')
    parser.add_argument('--limit', type=int, help='Limit rows to train/test/predict')
    parser.add_argument('--glove_path', type=str, help='Path to GLOVE embeddings to load')
    parser.add_argument('--preload_data', action='store_true',
                        help='Eagerly loads training and dev data. If CUDA selected, loads '
                             'into GPU memory.')

    return parser.parse_args()


@toolz.memoize
def get_tokenizer(token_regex):
    return RegexpTokenizer(token_regex)


def word_to_idx(vocab_size, token):
    """ Applies hashing trick to turn a word into its embedding idx
    :param str token: tokenized word
    :rtype: int
    """
    return (mmh3.hash(token) % (vocab_size - 1)) + 1


def transform_texts(settings, texts):
    """ Transforms texts based on provided args """
    if settings.char_rnn:
        raise NotImplementedError
    else:
        tensor = torch.zeros(settings.msg_len, len(texts)).type(torch.LongTensor)
        tokenizer = get_tokenizer(settings.token_regex)
        for row_idx, text in enumerate(texts):
            for col_idx, token in enumerate(toolz.take(settings.msg_len, tokenizer.tokenize(text))):
                tensor[col_idx, row_idx] = word_to_idx(settings.vocab_size, token)
    return Variable(tensor)
    return Variable(tensor.cuda()) if settings.cuda else Variable(tensor)


def transform_classes(settings, classes):
    """ Transforms classes based on provided args """
    class_dict = {cls: idx for idx, cls in enumerate(settings.classes)}
    torch_classes = torch.LongTensor([class_dict[cls] for cls in classes])
    return Variable(torch_classes)
    return Variable(torch_classes.cuda()) if settings.cuda else Variable(torch_classes)


def train_batch_iter(settings):
    yield from batch_iter_from_path(settings, settings.input_path)


def dev_batch_iter(settings):
    yield from batch_iter_from_path(settings, settings.validate_path)


def batch_iter_from_path(settings, path):
    """ Loads, transforms, and yields batches for training/testing/prediction """
    chunk_iter = iter(pandas.read_csv(path, chunksize=settings.batch_size, header=None,
                                      nrows=settings.get('limit')))
    while True:
        try:
            chunk = next(chunk_iter)
            real_chunk = chunk.dropna(axis=0)
            yield transform_texts(settings, real_chunk.loc[:, 0].values), \
                  transform_classes(settings, real_chunk.loc[:, 1].values)
        except pandas.errors.ParserError:
            pass


def predict_batch_iter(settings):
    chunk_iter = iter(pandas.read_csv(settings.input_path, chunksize=settings.batch_size,
                                      header=None, nrows=settings.get('limit')))
    while True:
        try:
            chunk = next(chunk_iter)
            real_chunk = chunk.dropna(axis=0)
            yield transform_texts(settings, real_chunk.loc[:, 0].values)
        except pandas.errors.ParserError:
            pass


def build_model(settings):
    """ Builds model based on arguments provided """
    if settings.char_rnn:
        raise NotImplementedError
    else:
        return WordRnn(settings)


def count_classes(settings):
    """ Infers class weights from provided training data """
    with open(settings.input_path) as infile:
        rows = toolz.take(settings.get('limit', 1000000000), csv.reader(infile))
        counts = Counter(row[1] for row in rows)
    return counts


def train(args):
    """ Trains RNN based on provided arguments """
    settings = Settings.from_args(args)
    model = build_model(settings)
    if args.continued:
        model = WordRnn(settings)
        model.load_state_dict(torch.load(args.model_path + '.bin'))
        logging.info('Model loaded from {}, continuing training'.format(settings.model_path))

    logging.info('Counting classes...')
    class_counts = count_classes(settings)
    if settings.class_weights:
        logging.info('Class weights specified: {}'.format(settings.class_weights))
        class_weights = torch.FloatTensor(settings.class_weights)
    else:
        class_weights = torch.FloatTensor([sum(class_counts.values()) / class_counts[c]
                                           for c in settings.classes])
        assert sum(class_counts.values()) > 0, \
            "Didn't find any examples of any classes (input iterator was empty)"
        logging.info('Inferred class weights: {}'.format(class_weights))
        settings.defaults['class_weights'] = list(class_weights)

    if settings.cuda:
        class_weights = class_weights.cuda()

    if settings.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    elif settings.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=settings.learning_rate)
    else:
        raise RuntimeError('Invalid optim value provided: {}'.format(settings.optimizer))

    if settings.loss_fn == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    elif settings.loss_fn == 'NLL':
        criterion = nn.NLLLoss(weight=class_weights)
    else:
        raise RuntimeError('Invalid loss value provided: {}'.format(settings.loss_fn))

    if COMET_API_KEY:
        assert COMET_PROJECT is not None, 'You must specify a comet project to use if providing' \
                                          ' COMET_API_KEY environment variable.'
        comet_experiment = Experiment(api_key=COMET_API_KEY, project_name=COMET_PROJECT,
                                      log_code=COMET_LOG_CODE)
        comet_experiment.log_multiple_params(settings.to_comet_hparams())
        comet_experiment.log_dataset_hash(open(settings.input_path).read())
    else:
        comet_experiment = MagicMock()

    try:
        if settings.preload_data:
            train_batches = list(tqdm(train_batch_iter(settings), 
                                      desc='Loading and transforming train batches...', 
                                      total=int(sum(class_counts.values())/settings.batch_size)))
            dev_batches = list(tqdm(dev_batch_iter(settings), 
                                    desc='Loading and transforming dev batches...')) \
                if settings.validate_path else None
        else:
            train_batches = None
            dev_batches = None
        for epoch in range(settings.epochs):
            model.train()
            train_epoch(settings, model, criterion, optimizer, epoch, comet_experiment,
                        class_counts, train_batches)
            if settings.get('validate_path'):
                model.eval()
                score_model(settings, model, criterion, epoch, comet_experiment, dev_batches)
            torch.save(model.state_dict(), settings.model_path + '.bin')
            settings.save(settings.model_path + '.toml')
            logging.info('Model saved at {}'.format(settings.model_path))
    except KeyboardInterrupt:
        pass


def train_epoch(settings, model, criterion, optimizer, epoch, comet_experiment, class_counts,
                train_batches):
    """ Trains a single epoch """
    comet_experiment.log_current_epoch(epoch)
    loss_queue = deque(maxlen=100)
    all_losses = []
    correct = 0
    seen = 0
    epoch_period = 1
    started = datetime.now()

    progress = tqdm(desc='Epoch {}'.format(epoch), total=sum(class_counts.values())) \
        if settings.verbose == 1 else MagicMock()

    for step, (batch_x, batch_y) in enumerate(train_batches or train_batch_iter(settings)):
        if settings.cuda:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
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
    if settings.verbose > 0:
        logging.info('Epoch: {}\tTrain accuracy: {:.3f}\tTrain loss: {:.3f}\tTrain rate: {:.3f}'
                     '\tTotal seconds: {}'.format(
            epoch, correct/seen, sum(all_losses) / len(all_losses), seen / epoch_period,
            epoch_period))


def train_batch(model, criterion, optimizer, x, y):
    """ Handles a single batch """
    model.zero_grad()
    output = model(x)

    loss = criterion(output, y)
    loss.backward()
    if model.gradient_clip:
        nn.utils.clip_grad_norm(model.parameters(), model.gradient_clip)
    optimizer.step()

    return output, loss.data[0]


def score_model(settings, model, criterion, epoch, comet_experiment, dev_batches):
    """ Score model and update comet """
    losses = []
    correct = 0
    seen = 0
    started = datetime.now()
    for batch_x, batch_y in (dev_batches or dev_batch_iter(settings)):
        if settings.cuda:
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
        model.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        losses.append(loss.data[0])
        seen += batch_x.size(1)
        correct += float(sum((output.max(1)[1] == batch_y).data.cpu().numpy()))

    period = (datetime.now() - started).total_seconds()
    comet_experiment.log_metric('dev_loss', sum(losses) / len(losses))
    comet_experiment.log_metric('dev_acc', correct / seen)
    if settings.verbose > 0:
        logging.info('Epoch: {}\t  Dev accuracy: {:.3f}\t  Dev loss: {:.3f}, Scored/sec: {}'.format(
            epoch, correct/seen, sum(losses) / len(losses)), seen / period)


def predict_batch(model, batch):
    if model.cuda:
        batch = batch.cuda()
    model.zero_grad()
    output = model(batch)
    return output.cpu() if model.cuda else output


def stdout_predict(args):
    args.context_dropout = 0
    args.embed_dropout = 0
    settings, model = load_settings_and_model(args.model_path, args)

    for batch_x in predict_batch_iter(settings):
        output = predict_batch(model, batch_x).data
        if settings.cuda:
            output = output.cpu()
        print('\n'.join(map(lambda idx: settings.classes[idx], output.max(1)[1].numpy())))


def predict(settings, model, texts):
    """ Predict classifications with the provided model and settings.  Returns iter of most likely
        classes.
    :param Settings settings:
    :param WordRnn model:
    :param list texts: Texts to predict for
    :rtype: iter
    """
    for batch in toolz.partition_all(settings.batch_size, texts):
        output = predict_batch(model, transform_texts(settings, batch)).data
        yield from map(lambda idx: settings.classes[idx], output.max(1)[1].numpy())


def main():
    args = parse_args()

    logging.basicConfig(format='%(levelname)s:%(asctime)s.%(msecs)03d [%(threadName)s] - %(message)s',
                        datefmt='%Y-%m-%d,%H:%M:%S',
                        filename=os.environ.get('LOG_PATH'),
                        level=getattr(logging, os.environ.get('LOG_LEVEL', 'INFO')))

    if args.mode == 'train':
        train(args)
    elif args.mode == 'predict':
        stdout_predict(args)
    elif args.mode == 'predict2':
        settings, model = load_settings_and_model('model')
        print(list(predict(settings, model, ['This is great!', 'Sometimes things just don\'t work out.'])))


if __name__ == '__main__':
    main()

