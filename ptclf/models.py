"""  """
import logging
import sqlite3

import numpy
import pandas
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

from ptclf.tokenizer import get_tokenizer
from ptclf.util import progress, get_classes


def build_model(settings):
    """ Builds model based on arguments provided """
    if settings.char_rnn:
        raise NotImplementedError
    else:
        return WordRnn(settings)


class WordRnn(nn.Module):
    def __init__(self, settings):
        """ Build model from settings
        :param Settings settings: Settings that describe the model to build
        """
        super(WordRnn, self).__init__()

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
            logging.debug('CUDA selected, changing components to CUDA...')
            self.switch_to_gpu()

        logging.debug('Initializing weights...')
        self.init_weights()
        logging.debug('Done building model.')

    def switch_to_cpu(self):
        self.embedding.cpu()
        self.rnn.cpu()
        self.dense.cpu()

    def switch_to_gpu(self):
        self.embedding.cuda()
        self.rnn.cuda()
        self.dense.cuda()

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
        torch_data = torch.load(settings.model_path + '.bin', map_location=None)
        model.load_state_dict(torch_data)
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


def load_embedding_weights_flat(settings):
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
                    weights[idx, :] = numpy.array(splits[1:settings.embed_dim + 1], dtype=float)
                except ValueError:
                    logging.error('Failed to convert the following line:\n{}'.format(splits))
                    raise
                remaining.remove(idx)
            if len(remaining) == 0:
                break
    if remaining:
        logging.debug('Filling remaining {} vocab words with random embeddings.'.format(
            len(remaining)))
        for idx in remaining:
            weights[idx, :] = numpy.random.rand(settings.embed_dim)
    return weights


def load_embedding_weights_sqlite(settings):
    sqlite_con = sqlite3.connect(settings.glove_path)
    tokenizer = get_tokenizer(settings)
    embeddings_query = """select * from vectors where token in ({})""".format(', '.join(
        "'{}'".format(token) for token in tokenizer.word_index if "'" not in token))
    weights = numpy.random.randn(settings.vocab_size, settings.embed_dim)
    for token, *rest in sqlite_con.execute(embeddings_query):
        weights[tokenizer.word_to_idx(token) - 1, :] = rest
    # df = pandas.read_sql(embeddings_query, sqlite_con)
    # for token, idx in tokenizer.word_index.items():
    #     if token in df.index:
    #         weights[idx, :] = df.loc[token].values
    #     else:
    #         weights[idx, :] = numpy.random.rand(settings.embed_dim)
    return weights

def load_embedding_weights(settings):
    """ Load embeddings weights
    :param str path: path to load embedding weights from
    :param Tokenizer tokenizer:
    :rtype: torch.FloatTensor
    """
    logging.debug('Loading embeddings from {}'.format(settings.glove_path))
    if settings.glove_path.endswith('.sqlite'):
        weights = load_embedding_weights_sqlite(settings)
    else:
        weights = load_embedding_weights_flat(settings)
    logging.debug('Done loading embedding weights.')
    return torch.from_numpy(weights)
