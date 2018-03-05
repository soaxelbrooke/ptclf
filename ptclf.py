#!/usr/bin/env python3.6
import json
import os
import re
import sqlite3
from copy import deepcopy
from datetime import datetime
from subprocess import check_output
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
import logging
import time

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.backends import cudnn

cudnn.benchmark = True

import toml

from sklearn import metrics
from collections import deque, OrderedDict
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
        self.num_classes = len(get_classes(settings))
        self.bidirectional = settings.bidirectional
        self.loss_fn = settings.loss_fn
        self.context_mode = settings.context_mode

        self.bidir_factor = 1 + int(self.bidirectional)
        if self.context_mode == 'attention':
            assert self.context_size * self.bidir_factor == self.embed_size, \
                'RNN output size must equal embed size for attention context_mode, got RNN output' \
                'size of {} and embed size of {}'.format(self.context_size * self.bidir_factor,
                                                         self.embed_size)

        self.embedding = nn.Embedding(self.input_size, self.embed_size, padding_idx=0)
        if settings.glove_path and not settings.continued:
            self.embedding.weight.data.copy_(load_embedding_weights(settings))

        self.embed_dropout = nn.Dropout2d(settings.embed_dropout)

        rnn_hidden_shape = (settings.rnn_layers * self.bidir_factor, 1, self.context_size)

        if self.rnn_kind == 'gru':
            if settings.learn_rnn_init:
                self.rnn_init_h = nn.Parameter(
                    torch.randn(*rnn_hidden_shape).type(torch.FloatTensor), requires_grad=True)
            else:
                self.rnn_init_h = Variable(torch.zeros(
                    self.rnn_layers * (1 + int(self.bidirectional)), 1, self.context_size))
            self.rnn = nn.GRU(self.embed_size, self.context_size, self.rnn_layers,
                              bidirectional=self.bidirectional)
        elif self.rnn_kind == 'lstm':
            if settings.learn_rnn_init:
                self.rnn_init_h1 = nn.Parameter(
                    torch.randn(*rnn_hidden_shape).type(torch.FloatTensor), requires_grad=True)
                self.rnn_init_h2 = nn.Parameter(
                    torch.randn(*rnn_hidden_shape).type(torch.FloatTensor), requires_grad=True)
            else:
                self.rnn_init_h1 = Variable(torch.zeros(
                    self.rnn_layers * (1 + int(self.bidirectional)), 1, self.context_size))
                self.rnn_init_h2 = Variable(torch.zeros(
                    self.rnn_layers * (1 + int(self.bidirectional)), 1, self.context_size))
            self.rnn = nn.LSTM(self.embed_size, self.context_size, self.rnn_layers,
                               bidirectional=self.bidirectional)
        else:
            raise RuntimeError('Got invalid rnn type: {}'.format(self.rnn_kind))

        self.rnn_dropout = nn.Dropout2d(settings.context_dropout)
        if self.context_mode == 'attention' or self.context_mode == 'maxavg':
            self.dense = nn.Linear(self.context_size * self.bidir_factor * 2, self.num_classes)
        else:
            self.dense = nn.Linear(self.context_size * self.bidir_factor, self.num_classes)

        if self.cuda:
            logging.info('CUDA selected, changing components to CUDA...')
            self.embedding.cuda()
            self.rnn.cuda()
            self.dense.cuda()

        logging.info('Initializing weights...')
        self.init_weights()

    def attend_to(self, context: torch.FloatTensor, embedded: torch.FloatTensor, smooth=True):
        similarities = torch.bmm(
            embedded.permute(1, 0, 2), context[-1:].permute(1, 2, 0)).sigmoid()
        alpha = similarities / torch.sum(similarities, 1, keepdim=True).expand_as(similarities)
        return torch.sum(alpha.permute(1, 0, 2) * embedded, 0)

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
        rnn_context, hidden = self.rnn(embedded, hidden_state)
        rnn_context = self.rnn_dropout(rnn_context)
        if self.context_mode == 'attention':
            context = torch.cat([self.attend_to(rnn_context, embedded), rnn_context[-1]], 1)
        elif self.context_mode == 'last':
            context = rnn_context[-1]
        elif self.context_mode == 'maxavg':
            context = torch.cat([torch.max(rnn_context, 0)[0], torch.mean(rnn_context, 0)], 1)
        else:
            context = torch.max(rnn_context, 0)[0]
        # context shape: (msg_len, batch_size, context_size)
        # hidden shape: (rnn_depth, batch_size, context_size)
        dense = self.dense(context)
        # dense shape: (batch_size, num_classes)
        if self.loss_fn == 'NLL' or self.loss_fn == 'CrossEntropy':
            class_probs = F.softmax(dense, dim=1)
        else:
            class_probs = F.sigmoid(dense)
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


def load_embedding_weights(settings):
    """ Load embeddings weights
    :param str path: path to load embedding weights from
    :param Tokenizer tokenizer:
    :rtype: torch.FloatTensor
    """
    logging.info('Loading embeddings from {}'.format(settings.glove_path))
    remaining = set(range(1, settings.vocab_size))
    weights = numpy.zeros((settings.vocab_size, settings.embed_dim), dtype=float)
    bar = progress(settings, desc='Loading weights...', total=settings.vocab_size)
    tokenizer = get_tokenizer(settings)
    with open(settings.glove_path, encoding='utf-8') as infile:
        for line in infile:
            if len(line) < 50:
                # skip header lines, etc - makes it compatible with fasttext vectors as well
                continue
            splits = line.strip('\n').split(' ')
            idx = tokenizer.word_to_idx(splits[0])
            if idx in remaining:
                bar.update(1)
                try:
                    weights[idx, :] = numpy.array(splits[1:settings.embed_dim+1], dtype=float)
                except ValueError:
                    logging.error('Failed to convert the following line:\n{}'.format(splits))
                    raise
                remaining.remove(idx)
            if len(remaining) == 0:
                break
    if remaining:
        logging.info('Filling remaining {} vocab words with random embeddings.'.format(
            len(remaining)))
        for idx in remaining:
            weights[idx, :] = numpy.random.rand(settings.embed_dim)
    logging.info('Done loading embedding weights.')
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


def progress(settings, iterator=None, desc=None, total=None):
    if settings.verbose != 1:
        if iterator is None:
            return MagicMock()
        else:
            return iterator
    if iterator is None:
        return tqdm(desc=desc, total=total)
    else:
        return tqdm(iterator, desc=desc, total=total)


def env(name, transform):
    var = os.environ.get(name)
    if var is not None:
        return transform(var)


def env_flag(name):
    return env(name, lambda s: s.lower() == 'true')


@toolz.memoize
def get_classes(settings):
    """ Infer classes from the provided settings
    :param Settings settings: The settings for which to infer the model classes
    :rtype: list
    """
    if settings.classes is None:
        df = pandas.read_csv(settings.input_path, nrows=1)
        settings.model_settings['classes'] = list(df.columns[1:].values)
    return settings.classes


class Settings:
    model_param_names = {'id', 'created_at', 'rnn', 'rnn_layers', 'char_rnn', 'bidirectional',
                         'classes', 'vocab_size', 'msg_len', 'context_dim', 'embed_dim',
                         'learn_rnn_init', 'context_mode', 'no_lowercase', 'filter_chars'}
    default_names = {'batch_size', 'epochs', 'cuda', 'learning_rate', 'optimizer', 'loss_fn',
                     'embed_dropout', 'context_dropout', 'token_regex', 'class_weights',
                     'gradient_clip', 'learn_class_weights'}
    transient_names = {'input_path', 'validate_path', 'verbose', 'limit', 'glove_path',
                       'model_path', 'preload_data', 'epoch_shell_callback', 'continued'}
    comet_hparam_names = ['rnn', 'rnn_layers', 'char_rnn', 'bidirectional', 'classes', 'vocab_size',
                          'msg_len', 'context_dim', 'embed_dim', 'batch_size', 'epochs', 'cuda',
                          'learning_rate', 'optimizer', 'loss_fn', 'embed_dropout',
                          'context_dropout', 'token_regex', 'class_weights', 'learn_rnn_init',
                          'context_mode']

    # Default values for if no setting is provided for given parameter
    model_param_defaults = {
        'rnn': 'gru', 'rnn_layers': 2, 'bidirectional': True, 'char_rnn': False, 'vocab_size': 1024,
        'msg_len': 40, 'context_dim': 32, 'embed_dim': 50, 'learn_rnn_init': False,
        'context_mode': 'maxavg', 'no_lowercase': False, 'filterchars': '',
    }

    default_defaults = {
        'epochs': 1, 'batch_size': 16, 'learning_rate': 0.005, 'optimizer': 'adam',
        'loss_fn': 'CrossEntropy', 'embed_dropout': 0.3, 'context_dropout': 0.3,
        'token_regex': r'\w+|\$[\d\.]+|\S+', 'gradient_clip': None, 'learn_class_weights': False,
    }

    transient_defaults = {'verbose': 1, 'glove_path': None, 'preload_data': False,
                          'epoch_shell_callback': None, 'validate_path': None}

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

    def __hash__(self):
        return hash(self.to_toml())

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
        with open(path, encoding='utf-8') as infile:
            settings_dict = toml.load(infile)
        settings = Settings(settings_dict['model'], settings_dict.get('defaults', {}), {})
        settings.transients['model_path'] = path[:-5]
        return settings

    def save(self, path):
        """ Save TOML settings to provided path.
        :param str path: Path to save settings to.
        :rtype: NoneType
        """
        with open(path, 'w', encoding='utf-8') as outfile:
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

    def copy(self):
        return Settings(deepcopy(self.model_settings), deepcopy(self.defaults),
                        deepcopy(self.transients))

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
    parser.add_argument('--hyperopt_spec', type=str)

    parser.add_argument('--learning_rate', type=float, default=env('LEARNING_RATE', float))
    parser.add_argument('--gradient_clip', type=float, default=env('GRADIENT_CLIP', float))
    parser.add_argument('--optimizer', type=str, default=env('OPTIMIZER', str),
                        help='One of {sgd, adam}')
    parser.add_argument('--loss_fn', type=str, default=env('LOSS_FN',
                                                           str))  # TODO add help details for other loss functions
    parser.add_argument('--embed_dropout', type=float, default=env('EMBED_DROPOUT', float),
                        help='Dropout used for embedding layer')
    parser.add_argument('--context_dropout', type=float, default=env('CONTEXT_DROPOUT', float),
                        help='Dropout used for RNN output')

    parser.add_argument('--rnn', type=str,
                        help='Type of RNN used - one of {gru, lstm}')
    parser.add_argument('--rnn_layers', type=int, default=env('RNN_LAYERS', int),
                        help='Number of RNN layers.  Best at 1 or 2 for word-level models, deeper '
                             'is more valuable for character-level models.')
    parser.add_argument('--bidirectional', action='store_true', default=env_flag('BIDIRECTIONAL'),
                        help='If set, RNN is bidirectional')
    parser.add_argument('--learn_rnn_init', action='store_true', default=env_flag('LEARN_RNN_INIT'),
                        help='Learn RNN initial state (default inits to tensor of 0s)')
    parser.add_argument('--context_mode', type=str, default=env('CONTEXT_MODE', str),
                        help='What goes between RNN output and dense layer, one of {attention, max,'
                             ' maxavg, last}.  Default is maxavg.')

    parser.add_argument('--char_rnn', action='store_true',
                        help='Use a character RNN instead of word RNN')
    parser.add_argument('-v', '--vocab_size', type=int, default=env('VOCAB_SIZE', int),
                        help='Vocab size')
    parser.add_argument('--msg_len', type=int, default=env('MSG_LEN', int),
                        help='Maximum length of text in tokens')
    parser.add_argument('-c', '--context_dim', type=int, default=env('CONTEXT_DIM', int),
                        help='Dimension of the RNN context vector')
    parser.add_argument('-e', '--embed_dim', type=int, default=env('EMBED_DIM', int),
                        help='Dimension of the embedding (only used for word-RNN). Larger is better'
                             ' for more abstract tasks like sentiment classification, and smaller '
                             'is better for more syntactic tasks like POS-tagging.')
    parser.add_argument('--token_regex', type=str,
                        default=env('TOKEN_REGEX', str),
                        help='Regexp pattern to tokenize with')
    parser.add_argument('--no_lowercase', action='store_true', default=env_flag('NO_LOWERCASE'),
                        help='Stops text from being lowercased in training and prediction.')
    parser.add_argument('--filter_chars', type=str, default='',
                        help='Removes all specified characters from input text')

    parser.add_argument('--classes', type=str,
                        help='Comma separated list of classes to predict')
    parser.add_argument('--learn_class_weights', action='store_true',
                        default=env_flag('LEARN_CLASS_WEIGHTS'),
                        help='Learnings class weights from data automatically.  Helps in cases '
                             'where class imbalance is a problem.')
    parser.add_argument('--class_weights', type=str,
                        default=env('CLASS_WEIGHTS', lambda s: s),
                        help='Comma separated list of class weights')

    parser.add_argument('--continued', action='store_true', help='Continue training')
    parser.add_argument('--limit', type=int, help='Limit rows to train/test/predict')
    parser.add_argument('--glove_path', type=str, help='Path to GLOVE embeddings to load')
    parser.add_argument('--preload_data', action='store_true',
                        help='Eagerly loads training and dev data. If CUDA selected, loads '
                             'into GPU memory.')
    parser.add_argument('--epoch_shell_callback', type=str,
                        help='Shell command executed after each epoch (after model save).',
                        default=env('EPOCH_SHELL_CALLBACK', str))
    parser.add_argument('--predict_top', action='store_true')

    return parser.parse_args()


class Tokenizer(object):
    """Text tokenization utility class.
    This class allows to vectorize a text corpus, by turning each
    text into either a sequence of integers (each integer being the index
    of a token in a dictionary) or into a vector where the coefficient
    for each token could be binary, based on word count, based on tf-idf...
    # Arguments
        num_words: the maximum number of words to keep, based
            on word frequency. Only the most common `num_words` words will
            be kept.
        filters: a string where each element is a character that will be
            filtered from the texts. The default is all punctuation, plus
            tabs and line breaks, minus the `'` character.
        lower: boolean. Whether to convert the texts to lowercase.
        split: character or string to use for token splitting.
        char_level: if True, every character will be treated as a token.
        oov_token: if given, it will be added to word_index and used to
            replace out-of-vocabulary words during text_to_sequence calls
    By default, all punctuation is removed, turning the texts into
    space-separated sequences of words
    (words maybe include the `'` character). These sequences are then
    split into lists of tokens. They will then be indexed or vectorized.
    `0` is a reserved index that won't be assigned to any word.
    """

    def __init__(self, vocab_size=None, msg_len=100, filters='', lower=True,
                 split=' ', char_rnn=False, oov_token=None):
        self.word_counts = OrderedDict()
        self.word_docs = {}
        self.msg_len = msg_len
        self.filters = filters
        self.re_filter = re.compile('[' + re.escape(filters) + ']') if filters else None
        self.split = split
        self.re_split = re.compile(split)
        self.lower = lower
        self.vocab_size = vocab_size
        self.document_count = 0
        self.char_rnn = char_rnn
        self.oov_token = oov_token
        self.index_docs = {}
        self.word_index = {}

    @classmethod
    def from_json(cls, toml_text):
        """ Loads tokenizer and vocab from provided toml text """
        jobj = json.loads(toml_text)
        meta = jobj['meta']
        tokenizer = cls(meta['vocab_size'], meta['msg_len'], meta['filters'], meta['lower'],
                        meta['split'], meta['char_rnn'], meta.get('oov_token'))
        tokenizer.document_count = meta['document_count']
        tokenizer.word_counts = OrderedDict(jobj['word_counts'])
        tokenizer.word_docs = jobj['word_docs']
        tokenizer.word_index = jobj['word_index']
        tokenizer.index_docs = {int(k): v for k, v in jobj['index_docs'].items()}
        return tokenizer

    @classmethod
    def load(cls, path):
        """ Loads JSON serialized tokenizer from path """
        logging.info('Loading tokenizer from {}...'.format(path))
        with open(path) as infile:
            return cls.from_json(infile.read())

    def to_json(self):
        """ Serializes tokenizer and vocab to toml """
        jobj = {
            'meta': {
                'msg_len': self.msg_len, 'filters': self.filters, 'split': self.split,
                'lower': self.lower, 'vocab_size': self.vocab_size, 'char_rnn': self.char_rnn,
                'document_count': self.document_count, 'oov_token': self.oov_token,
            },
            'word_counts': dict(self.word_counts),
            'word_docs': self.word_docs,
            'index_docs': {str(k): v for k, v in self.index_docs.items()},
            'word_index': self.word_index,
        }
        return json.dumps(jobj)

    def save(self, path):
        """ Save JSON version of tokenizer to path """
        logging.info('Saving tokenizer to {}...'.format(path))
        json_text = self.to_json()
        with open(path, 'w') as outfile:
            outfile.write(json_text)

    def text_to_word_sequence(self, text):
        if self.lower:
            text = text.lower()

        if self.re_filter:
            text = self.re_filter.sub('', text)

        return [tok for tok in self.re_split.findall(text) if tok]

    def word_to_idx(self, token):
        if self.lower:
            token = token.lower()
        if self.re_filter:
            token = self.re_filter.sub('', token)
        return self.word_index.get(token)

    def fit_on_texts(self, texts):
        """Updates internal vocabulary based on a list of texts.
        In the case where texts contains lists, we assume each entry of the lists
        to be a token.
        Required before using `texts_to_sequences` or `texts_to_matrix`.
        # Arguments
            texts: can be a list of strings,
                a generator of strings (for memory-efficiency),
                or a list of list of strings.
        """
        for text in texts:
            self.document_count += 1
            if self.char_rnn or isinstance(text, list):
                seq = text
            else:
                seq = self.text_to_word_sequence(text)
            for w in seq:
                if w in self.word_counts:
                    self.word_counts[w] += 1
                else:
                    self.word_counts[w] = 1
            for w in set(seq):
                if w in self.word_docs:
                    self.word_docs[w] += 1
                else:
                    self.word_docs[w] = 1

        wcounts = list(self.word_counts.items())
        wcounts.sort(key=lambda x: x[1], reverse=True)
        sorted_voc = [wc[0] for wc in wcounts]
        # note that index 0 is reserved, never assigned to an existing word
        self.word_index = dict(list(zip(sorted_voc, list(range(1, len(sorted_voc) + 1)))))

        if self.oov_token is not None:
            i = self.word_index.get(self.oov_token)
            if i is None:
                self.word_index[self.oov_token] = len(self.word_index) + 1

        for w, c in list(self.word_docs.items()):
            self.index_docs[self.word_index[w]] = c

    def fit_on_sequences(self, sequences):
        """Updates internal vocabulary based on a list of sequences.
        Required before using `sequences_to_matrix`
        (if `fit_on_texts` was never called).
        # Arguments
            sequences: A list of sequence.
                A "sequence" is a list of integer word indices.
        """
        self.document_count += len(sequences)
        for seq in sequences:
            seq = set(seq)
            for i in seq:
                if i not in self.index_docs:
                    self.index_docs[i] = 1
                else:
                    self.index_docs[i] += 1

    def texts_to_sequences(self, texts):
        """Transforms each text in texts in a sequence of integers.
        Only top "num_words" most frequent words will be taken into account.
        Only words known by the tokenizer will be taken into account.
        # Arguments
            texts: A list of texts (strings).
        # Returns
            A list of sequences.
        """
        res = []
        for vect in self.texts_to_sequences_generator(texts):
            res.append(vect)
        return res

    def texts_to_sequences_generator(self, texts):
        """Transforms each text in `texts` in a sequence of integers.
        Each item in texts can also be a list, in which case we assume each item of that list
        to be a token.
        Only top "num_words" most frequent words will be taken into account.
        Only words known by the tokenizer will be taken into account.
        # Arguments
            texts: A list of texts (strings).
        # Yields
            Yields individual sequences.
        """
        num_words = self.vocab_size
        for text in texts:
            if self.char_rnn or isinstance(text, list):
                seq = text
            else:
                seq = self.text_to_word_sequence(text)
            vect = []
            for w in seq:
                i = self.word_index.get(w)
                if i is not None:
                    if num_words and i >= num_words:
                        continue
                    else:
                        vect.append(i)
                elif self.oov_token is not None:
                    i = self.word_index.get(self.oov_token)
                    if i is not None:
                        vect.append(i)
            yield vect

    def transform_texts(self, texts):
        sequences = self.texts_to_sequences(texts)
        tensor = torch.zeros(self.msg_len, len(texts)).type(torch.LongTensor)
        for row_idx, sequence in enumerate(sequences):
            for col_idx, word_idx in enumerate(sequence[:self.msg_len]):
                tensor[col_idx, row_idx] = word_idx
        return Variable(tensor)


@toolz.memoize
def get_tokenizer(settings):
    tokenizer_path = settings.model_path + '.tokenizer.json'
    if os.path.isfile(tokenizer_path):
        tokenizer = Tokenizer.load(tokenizer_path)
    else:
        logging.info('Learning vocab...')
        tokenizer = Tokenizer(settings.vocab_size, settings.msg_len, settings.filter_chars,
                              not settings.no_lowercase, settings.token_regex, settings.char_rnn)
        texts = pandas.read_csv(settings.input_path).text.dropna()
        tokenizer.fit_on_texts(
            progress(settings, texts, desc='Learning vocab...', total=len(texts)))
        tokenizer.save(tokenizer_path)
    return tokenizer


def train_batch_iter(settings):
    yield from batch_iter_from_path(settings, settings.input_path)


def dev_batch_iter(settings):
    yield from batch_iter_from_path(settings, settings.validate_path)


def batch_iter_from_path(settings, path):
    """ Loads, transforms, and yields batches for training/testing/prediction """
    tokenizer = get_tokenizer(settings)
    chunk_iter = iter(pandas.read_csv(path, chunksize=settings.batch_size,
                                      nrows=settings.get('limit')))
    while True:
        try:
            chunk = next(chunk_iter)
            real_chunk = chunk.dropna(axis=0)
            # real_chunk = real_chunk[real_chunk[real_chunk.columns[1:]].sum(axis=1) > 0]
            classes = real_chunk[real_chunk.columns[1:]]
            if settings.loss_fn == 'CrossEntropy' or settings.loss_fn == 'NLL':
                classes = torch.LongTensor(classes.values.argmax(axis=1))
            else:
                classes = torch.FloatTensor(classes.values)
            yield tokenizer.transform_texts(real_chunk.text.values), \
                  Variable(classes)
        except pandas.errors.ParserError:
            pass


def predict_batch_iter(settings):
    tokenizer = get_tokenizer(settings)
    chunk_iter = iter(pandas.read_csv(settings.input_path, chunksize=settings.batch_size,
                                      header=None, nrows=settings.get('limit')))
    while True:
        try:
            chunk = next(chunk_iter)
            real_chunk = chunk.dropna(axis=0)
            yield tokenizer.transform_texts(real_chunk.loc[:, 0].values)
        except pandas.errors.ParserError:
            pass


def num_correct(output, batch_y):
    """ Calculate number of correct predictions """
    return float(sum((output.max(1)[1] == batch_y).data.cpu().numpy()))


def auroc(output, batch_y):
    """ Calculates the Area Under the Receiver Operator Characteristic curve """
    if output.is_cuda:
        output = output.cpu()
        batch_y = batch_y.cpu()
    aucrocs = []
    for class_idx in range(output.shape[1]):
        y = batch_y[:, class_idx].data.numpy()
        p = output[:, class_idx].data.numpy()
        if len(numpy.unique(y)) == 1:
            continue
        aucrocs.append(metrics.roc_auc_score(y, p))
    if len(aucrocs):
        return numpy.mean(aucrocs)
    else:
        return 0


def build_model(settings):
    """ Builds model based on arguments provided """
    if settings.char_rnn:
        raise NotImplementedError
    else:
        return WordRnn(settings)


def count_classes(settings):
    """ Infers class weights from provided training data """
    df = pandas.read_csv(settings.input_path)
    return df[df.columns[1:]].sum(axis=0).to_dict()


def hyperopt(args):
    from bayes_opt import BayesianOptimization

    settings = Settings.from_args(args)
    with open(args.hyperopt_spec) as infile:
        spec = toml.load(infile)

    def run_hyperopt_experiment(**kwargs):
        """ Creates a settings TOML file and trains a model using it """
        for k, v in kwargs.items():
            if 'dropout' not in k and isinstance(v, float):
                kwargs[k] = int(v)
        experiment_id = uuid4()
        logging.info('Running experiment {}...'.format(experiment_id))
        experiment_settings = settings.copy()
        experiment_settings.model_settings['id'] = experiment_id
        for key, value in kwargs.items():
            experiment_settings.model_settings[key] = value
        settings_path = 'model-{}.toml'.format(experiment_id)
        experiment_settings.save(settings_path)
        check_output(['python3.6', 'ptclf.py', 'train', '--verbose', '2', '-m', settings_path[:-5],
                      '-i', settings.input_path, '--validate_path', settings.validate_path])

        con = sqlite3.connect('experiments.sqlite')
        results = con.execute(
            "select dev_loss from metrics where experiment_id='{}' and dev_loss not null order by epoch desc limit 1".format(
                experiment_id))
        return list(results)[0][0]

    gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 2}
    bayes_opt = BayesianOptimization(run_hyperopt_experiment, spec['bounds'])
    # bayes_opt.explore(spec['points'])
    bayes_opt.maximize(n_iter=25, acq="ucb", kappa=5, **gp_params)


def train(args):
    """ Trains RNN based on provided arguments """
    settings = Settings.from_args(args)
    sle = SqliteExperiment(
        [('rnn', str), ('rnn_layers', int), ('char_rnn', bool), ('bidirectional', bool),
         ('classes', str), ('vocab_size', int), ('msg_len', int), ('context_dim', int),
         ('embed_dim', int), ('batch_size', int), ('epochs', int), ('cuda', bool),
         ('learning_rate', float), ('optimizer', str), ('loss_fn', str), ('embed_dropout', float),
         ('context_dropout', float), ('token_regex', str), ('learn_rnn_init', bool),
         ('context_mode', str)],
        [('loss', float), ('dev_loss', float), ('epoch', int), ('samples_seen', int),
         ('acc', float), ('dev_acc', float), ('train_per_second', float),
         ('score_per_second', float)],
        os.environ.get('EXPERIMENT_ID', settings.id))
    sle.log_hparams(settings.to_comet_hparams())
    model = build_model(settings)
    if args.continued:
        model = WordRnn(settings)
        model.load_state_dict(torch.load(args.model_path + '.bin'))
        logging.info('Model loaded from {}, continuing training'.format(settings.model_path))
    tokenizer = get_tokenizer(settings)

    if settings.class_weights:
        logging.info('Class weights specified: {}'.format(settings.class_weights))
        class_weights = torch.FloatTensor(settings.class_weights)
    elif settings.learn_class_weights:
        logging.info('Learning class weights...')
        class_counts = count_classes(settings)
        class_weights = torch.FloatTensor([sum(class_counts.values()) / class_counts[c]
                                           for c in get_classes(settings)])
        class_weights /= class_weights.min()
        assert sum(class_counts.values()) > 0, \
            "Didn't find any examples of any classes (input iterator was empty)"
        logging.info('Inferred class weights: {}'.format(class_weights))
        settings.defaults['class_weights'] = list(class_weights)
    else:
        class_weights = torch.FloatTensor([1.0] * len(get_classes(settings)))

    if settings.cuda:
        class_weights = class_weights.cuda()

    if settings.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    elif settings.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=settings.learning_rate)
    else:
        raise RuntimeError('Invalid optim value provided: {}'.format(settings.optimizer))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    if settings.loss_fn == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    elif settings.loss_fn == 'NLL':
        criterion = nn.NLLLoss(weight=class_weights)
    elif settings.loss_fn == 'MultiLabelMargin':
        criterion = nn.MultiLabelMarginLoss()
    elif settings.loss_fn == 'BCE':
        criterion = nn.BCELoss()
    else:
        raise RuntimeError('Invalid loss value provided: {}'.format(settings.loss_fn))

    if COMET_API_KEY:
        assert COMET_PROJECT is not None, 'You must specify a comet project to use if providing' \
                                          ' COMET_API_KEY environment variable.'
        comet_experiment = Experiment(api_key=COMET_API_KEY, project_name=COMET_PROJECT,
                                      log_code=COMET_LOG_CODE, parse_args=False)
        comet_experiment.log_multiple_params(settings.to_comet_hparams())
    else:
        comet_experiment = MagicMock()

    try:
        if settings.preload_data:
            train_batches = list(progress(
                settings, train_batch_iter(settings), desc='Loading train batches...',
                total=int(tokenizer.document_count / settings.batch_size)))

            dev_batches = list(progress(settings, dev_batch_iter(settings),
                                        desc='Loading dev batches...')) \
                if settings.validate_path else None
        else:
            train_batches = None
            dev_batches = None
        for epoch in range(settings.epochs):
            model.train()
            train_epoch(settings, model, criterion, optimizer, epoch, comet_experiment,
                        train_batches, sle)
            if settings.get('validate_path'):
                model.eval()
                val_loss = score_model(settings, model, criterion, epoch, comet_experiment,
                                       dev_batches, sle)
                scheduler.step(val_loss, epoch=epoch)
            torch.save(model.state_dict(), settings.model_path + '.bin')
            settings.save(settings.model_path + '.toml')
            logging.info('Model saved at {}'.format(settings.model_path))
            if settings.epoch_shell_callback:
                logging.info('Executing epoch callback: {}'.format(settings.epoch_shell_callback))
                check_output(settings.epoch_shell_callback, shell=True)
    except KeyboardInterrupt:
        pass


def train_epoch(settings, model, criterion, optimizer, epoch, comet_experiment, train_batches, sle):
    """ Trains a single epoch """
    tokenizer = get_tokenizer(settings)
    comet_experiment.log_current_epoch(epoch)
    loss_queue = deque(maxlen=100)
    all_losses = []
    accuracies = []
    seen = 0
    epoch_period = 1
    started = datetime.now()

    progress = tqdm(desc='Epoch {}'.format(epoch), total=tokenizer.document_count) \
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
        if settings.loss_fn != 'CrossEntropy' and settings.loss_fn != 'NLL':
            accuracies.append(auroc(output, batch_y))
        else:
            accuracies.append(num_correct(output, batch_y) / batch_x.shape[1])
        accuracy = numpy.mean(accuracies)

        sle.log_metrics(epoch, seen, {'loss': loss, 'epoch': epoch, 'samples_seen': seen,
                                      'acc': accuracy, 'train_per_second': seen / epoch_period})
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
            epoch, numpy.mean(accuracies), sum(all_losses) / len(all_losses), seen / epoch_period,
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


def score_model(settings, model, criterion, epoch, comet_experiment, dev_batches, sle):
    """ Score model and update comet """
    losses = []
    accuracies = []
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
        if settings.loss_fn != 'CrossEntropy' and settings.loss_fn != 'NLL':
            accuracies.append(auroc(output, batch_y))
        else:
            accuracies.append(num_correct(output, batch_y) / batch_x.shape[1])

    accuracy = numpy.mean(accuracies)
    period = (datetime.now() - started).total_seconds()
    mean_loss = sum(losses) / len(losses)
    sle.log_metrics(epoch, seen, {'dev_loss': mean_loss, 'epoch': epoch,
                                  'dev_acc': accuracy, 'score_per_second': seen / period},
                    force=True)
    comet_experiment.log_metric('dev_loss', mean_loss)
    comet_experiment.log_metric('dev_acc', accuracy)
    if settings.verbose > 0:
        logging.info('Epoch: {}\t  Dev accuracy: {:.3f}\t  Dev loss: {:.3f}, Scored/sec: {}'.format(
            epoch, accuracy, sum(losses) / len(losses), seen / period))
    return sum(losses) / len(losses)


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
        if args.predict_top:
            print('\n'.join(settings.classes[c] for c in output.max(1)[1]))
        else:
            for row in output.numpy().tolist():
                print(','.join(map(str, row)))


def predict(settings, model, texts):
    """ Predict classifications with the provided model and settings.  Returns iter of most likely
        classes.
    :param Settings settings:
    :param WordRnn model:
    :param list texts: Texts to predict for
    :rtype: iter
    """
    started = datetime.now()
    tokenizer = get_tokenizer(settings)
    for batch in toolz.partition_all(settings.batch_size, texts):
        output = predict_batch(model, tokenizer.transform_texts(batch)).data
        yield from map(lambda idx: settings.classes[idx], output.max(1)[1].numpy())
    logging.info('Made {} predictions in {} seconds.'.format(
        len(texts), (datetime.now() - started).total_seconds()))


class SqliteExperiment:
    def __init__(self, hparams, metrics, experiment_id=None):
        self.experiment_id = experiment_id or str(uuid4())
        self.hparams = hparams
        self.metrics = metrics
        self.metric_names = ['experiment_id', 'measured_at'] + [n for n, t in metrics]
        self.log_every = int(os.environ.get('LOG_EVERY', 10000))
        self.last_log = None
        self.last_epoch = None
        self.db = sqlite3.connect('experiments.sqlite')
        self.ensure_tables()

    def ensure_tables(self):
        """ Create tables for metrics and hyper params if they don't exist """
        self.db.execute('''
CREATE TABLE IF NOT EXISTS hparams (
  experiment_id text primary key,
  {}
)
        '''.format(self.to_sql_column_defs(self.hparams).strip(',')))

        self.db.execute('''
CREATE TABLE IF NOT EXISTS metrics (
  experiment_id text,
  measured_at int,
  {}
)
        '''.format(self.to_sql_column_defs(self.metrics).strip(',')))
        self.db.commit()

    @classmethod
    def to_sqlite_col_type(cls, col_type):
        return {
            int: 'integer',
            float: 'real',
            str: 'text',
            bool: 'integer',
        }[col_type]

    def to_sql_column_defs(self, spec):
        return ',\n'.join([
            '{} {}'.format(col_name, self.to_sqlite_col_type(col_type))
            for col_name, col_type in spec
        ]) + ','

    def log_hparams(self, hparams):
        hparam_values = [self.experiment_id] + [hparams[name] for name, _type in self.hparams]
        for idx, hparam in enumerate(hparam_values):
            if isinstance(hparam, list):
                hparam_values[idx] = ','.join(map(str, hparam))
        self.db.execute('''
insert into hparams values ({})
        '''.format(', '.join(['?'] * len(hparam_values))), hparam_values)
        self.db.commit()

    def should_log(self, epoch, step):
        should_log = False
        if (self.last_epoch is None or self.last_log is None) \
                or (self.last_epoch < epoch) \
                or ((self.last_log + self.log_every) < step):
            self.last_epoch = epoch
            self.last_log = step
            should_log = True
        return should_log

    def log_metrics(self, epoch, step, metrics, force=False):
        if not force and not self.should_log(epoch, step):
            return
        metric_values = [self.experiment_id, time.time()] + \
                        [metrics.get(name) for name, _type in self.metrics]
        self.db.execute(
            '''
insert into metrics ({}) values ({})
            '''.format(', '.join(self.metric_names), ', '.join(['?'] * len(metric_values))),
            metric_values)
        self.db.commit()


def main():
    args = parse_args()

    logging.basicConfig(
        format='%(levelname)s:%(asctime)s.%(msecs)03d [%(threadName)s] - %(message)s',
        datefmt='%Y-%m-%d,%H:%M:%S',
        filename=os.environ.get('LOG_PATH'),
        level=getattr(logging, os.environ.get('LOG_LEVEL', 'INFO')))

    if args.mode == 'train':
        train(args)
    elif args.mode == 'hyperopt':
        hyperopt(args)
    elif args.mode == 'predict':
        stdout_predict(args)
    elif args.mode == 'predict2':
        settings, model = load_settings_and_model('model')
        examples = [
            'This is great!',
            'Sometimes things just don\'t work out.',
            'Nothing to complain about!',
            'It defeinitely lives up to the hype!',
            'After everything people said, a definite disappointment.  Hopefully they do better next time.',
            'If that means no more fucking around with nvidia drivers on aws then sign me up.',
            'New and improved, and now completely unusable since update. Zestimate is under a heading "Home Value" It is fraudulent and misleading and needs to be eliminated.',
            'Spammy. When you share something instead of a URL it dumps half a page of ad text and app links. Then it signs your email up for reports on that property. Geez, forbid I share a listing with a friend... Going to just find something less spam filled.',
        ]
        results = list(predict(settings, model, examples))
        for example, result in zip(examples, results):
            print(example, '=>', result)


if __name__ == '__main__':
    main()
