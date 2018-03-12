""" Tokenizer class for learning, saving, and loading tokenizers (with vocabs) """

import json
import logging
import re
from collections import OrderedDict
from typing import Optional
import sqlite3
from sqlite3 import Connection


import pandas
import toolz
import torch
from torch.autograd import Variable

from ptclf.util import progress


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

    def __init__(self, vocab_size: Optional[int]=None, msg_len: int=100, filters: str='',
                 lower: bool=True, split: str=r'\w+|\$[\d\.]+|\S+', char_rnn: bool=False,
                 oov_token: Optional[str]=None, trim_after_fit: bool=True):
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
        self.trim_after_fit = trim_after_fit

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
    def load(cls, sqlite_con: Connection):
        """ Load tokenizer from the provided sqlite database """
        tok = cls(*next(iter(sqlite_con.execute('select * from tokenizer_config;'))))
        tok.word_index = dict(sqlite_con.execute('select * from tokenizer_word_indexes;'))
        return tok

    def save(self, sqlite_con: Connection):
        """ Save tokenizer to provided sqlite database """
        config_table_create = """
            create table if not exists tokenizer_config (
              vocab_size integer,
              msg_len integer,
              filters text,
              lower integer,
              split text,
              char_rnn integer,
              oov_token text,
              trim_after_fit integer
            );
        """
        config_insert = "insert into tokenizer_config values(?, ?, ?, ?, ?, ?, ?, ?);"
        word_indexes_table_create = """
            create table if not exists tokenizer_word_indexes (word text, idx integer);
        """
        word_indexes_insert = "insert into tokenizer_word_indexes VALUES (?, ?);"

        sqlite_con.execute(config_table_create)
        sqlite_con.execute(config_insert, [
            self.vocab_size, self.msg_len, self.filters, self.lower, self.split, self.char_rnn,
            self.oov_token, self.trim_after_fit,
        ])
        sqlite_con.execute(word_indexes_table_create)
        sqlite_con.executemany(word_indexes_insert, self.word_index.items())
        sqlite_con.commit()

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
        logging.debug('Counting documents and words...')
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

        logging.debug('Sorting word counts...')
        sorted_voc = [wc[0] for wc in sorted(self.word_counts.items(), key=lambda x: -x[1])]
        # note that index 0 is reserved, never assigned to an existing word
        logging.debug('Zipping vocab with word indexes...')
        self.word_index = dict(zip(sorted_voc, range(1, self.vocab_size + 1)))

        if self.oov_token is not None:
            i = self.word_index.get(self.oov_token)
            if i is None:
                self.word_index[self.oov_token] = len(self.word_index) + 1

        if self.trim_after_fit:
            self.trim_vocab()

        logging.debug('Done fitting tokenizer.')

    def trim_vocab(self):
        """ Trim vocab to vocab_size to save memory. Un-trainable afterwards. """
        logging.debug('Trimming vocab down to {} tokens...'.format(self.vocab_size))
        self.word_counts = {}
        self.word_docs = {}
        self.index_docs = {}

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
    sqlite_con = sqlite3.connect(settings.model_path + '.sqlite')
    try:
        tokenizer = Tokenizer.load(sqlite_con)
        if tokenizer.vocab_size == settings.vocab_size:
            return tokenizer
        logging.debug('Found tokenizer had wrong vocab size ({}), wanted {}.'.format(
            tokenizer.vocab_size, settings.vocab_size))
    except:
        pass
    logging.debug('Learning vocab...')
    tokenizer = Tokenizer(settings.vocab_size, settings.msg_len, settings.filter_chars,
                          not settings.no_lowercase, settings.token_regex, settings.char_rnn)
    texts = pandas.read_csv(settings.input_path).text.dropna()
    tokenizer.fit_on_texts(
        progress(settings, texts, desc='Learning vocab...', total=len(texts)))
    tokenizer.save(sqlite_con)
    return tokenizer

