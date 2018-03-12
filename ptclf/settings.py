"""
Settings parser, accessor, and save/load.  Settings can be specified when the model is created via
command line args or environment variables.  Once a model has been trained, the only settings that
can be overridden are batch size, shell callbacks, data preloading, and CUDA usage.  Settings are
never loaded in `train` mode.
"""
import csv
from copy import deepcopy
from datetime import datetime
from uuid import uuid4
import argparse

from sqlite3 import Connection

import os


def env(name, transform):
    var = os.environ.get(name)
    if var is not None:
        return transform(var)


def env_flag(name):
    return env(name, lambda s: s.lower() == 'true')


class Settings:
    _model_param_names = ['id', 'created_at', 'rnn', 'rnn_layers', 'char_rnn', 'bidirectional',
                          'classes', 'vocab_size', 'msg_len', 'context_dim', 'embed_dim',
                          'learn_rnn_init', 'context_mode', 'no_lowercase', 'filter_chars']
    _model_bools = {'char_rnn', 'bidirectional', 'learn_rnn_init', 'no_lowercase'}
    _default_names = ['batch_size', 'epochs', 'cuda', 'learning_rate', 'optimizer', 'loss_fn',
                      'embed_dropout', 'context_dropout', 'token_regex', 'class_weights',
                      'gradient_clip', 'learn_class_weights']
    _default_bools = {'cuda', 'learn_class_weights'}
    model_param_names = set(_model_param_names)
    default_names = set(_default_names)
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
        'context_mode': 'maxavg', 'no_lowercase': False, 'filter_chars': '',
    }

    default_defaults = {
        'epochs': 1, 'batch_size': 16, 'learning_rate': 0.005, 'optimizer': 'adam',
        'loss_fn': 'CrossEntropy', 'embed_dropout': 0.3, 'context_dropout': 0.3,
        'token_regex': r'\w+|\$[\d\.]+|\S+', 'gradient_clip': None, 'learn_class_weights': False,
        'cuda': False,
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
        return hash(self.to_sql_insert())

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

        if self.model_settings.get('classes') is None:
            with open(self.transients['input_path']) as infile:
                header = next(iter(csv.reader(infile)))
            self.model_settings['classes'] = [val for val in header if val != 'text']

        if self.defaults.get('class_weights') is None:
            self.defaults['class_weights'] = [1.0 for _ in self.model_settings['classes']]

    def to_sql_insert(self):
        all_columns = self._model_param_names + self._default_names
        placeholders = ', '.join(['?'] * len(all_columns))
        col_stmt = ', '.join(all_columns)
        insert_statement = "insert into settings ({}) values ({})".format(col_stmt, placeholders)
        values = []

        for name in self._model_param_names:
            if name == 'created_at':
                value = self.model_settings[name].timestamp()
            elif name == 'classes':
                value = ','.join(map(str, self.model_settings[name]))
            else:
                value = self.model_settings[name]
            values.append(value)

        for name in self._default_names:
            if name == 'class_weights':
                value = ','.join(map(str, self.defaults[name]))
            else:
                value = self.defaults[name]
            values.append(value)

        return insert_statement, tuple(values)

    @classmethod
    def load(cls, sqlite_con: Connection) -> 'Settings':
        """ Read settings from specified path """
        model_params_query = "select {} from settings".format(', '.join(cls._model_param_names))
        model_params = dict(zip(cls._model_param_names,
                                list(sqlite_con.execute(model_params_query))[0]))
        defaults_query = "select {} from settings".format(', '.join(cls._default_names))
        defaults = dict(zip(cls._default_names, list(sqlite_con.execute(defaults_query))[0]))

        model_params['created_at'] = datetime.fromtimestamp(model_params['created_at'])
        model_params['classes'] = model_params['classes'].split(',')
        defaults['class_weights'] = tuple(map(float, defaults['class_weights'].split(',')))

        for key in model_params:
            if key in cls._model_bools:
                model_params[key] = bool(model_params[key])

        for key in defaults:
            if key in cls._default_bools:
                defaults[key] = bool(defaults[key])

        settings = Settings(model_params, defaults, {})
        settings.try_defaults()
        return settings

    def save(self, sqlite_con: Connection):
        """ Save settings to the provided sqlite db. """
        table_create = """
            create table settings (
              id text, created_at integer, rnn text, rnn_layers integer, char_rnn integer, 
              bidirectional integer, classes text[], vocab_size integer, msg_len integer,
              context_dim integer, embed_dim integer, learn_rnn_init integer, context_mode text,
              no_lowercase integer, filter_chars text, batch_size integer, epochs integer, 
              cuda integer, learning_rate real, optimizer text, loss_fn text, embed_dropout real,
              context_dropout real, token_regex text, class_weights real[], gradient_clip real,
              learn_class_weights integer
            );
        """

        crs = sqlite_con.cursor()
        crs.execute('DROP TABLE IF EXISTS settings;')
        crs.execute(table_create)
        crs.execute(*self.to_sql_insert())
        sqlite_con.commit()

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
    """ Parses command line args and env vars and adds them to current settings. """
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
    parser.add_argument('--random_seed', type=int)

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
