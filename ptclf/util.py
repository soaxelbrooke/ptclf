""" Utility functions for ptclf """
import sqlite3

import numpy
from sklearn import metrics
from unittest.mock import MagicMock

import pandas
import toolz
import torch
from torch.autograd import Variable
from tqdm import tqdm

from ptclf.settings import Settings


@toolz.memoize
def get_classes(settings: Settings):
    """ Infer classes from the provided settings """
    if settings.classes is None:
        df = pandas.read_csv(settings.input_path, nrows=1)
        settings.model_settings['classes'] = list(df.columns[1:].values)
    return settings.classes


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


def count_classes(settings):
    """ Infers class weights from provided training data """
    df = pandas.read_csv(settings.input_path)
    return df[df.columns[1:]].sum(axis=0).to_dict()

def train_batch_iter(settings):
    yield from batch_iter_from_path(settings, settings.input_path)


def dev_batch_iter(settings):
    yield from batch_iter_from_path(settings, settings.validate_path)


def batch_iter_from_path(settings, path):
    """ Loads, transforms, and yields batches for training/testing/prediction """
    from ptclf.tokenizer import get_tokenizer
    tokenizer = get_tokenizer(settings)
    chunk_iter = iter(pandas.read_csv(path, chunksize=settings.batch_size,
                                      nrows=settings.get('limit')))
    while True:
        try:
            chunk = next(chunk_iter)
            real_chunk = chunk.dropna(axis=0)
            classes = real_chunk[real_chunk.columns[1:]]
            if settings.loss_fn == 'CrossEntropy' or settings.loss_fn == 'NLL':
                classes = torch.LongTensor(classes.values.argmax(axis=1))
            else:
                classes = torch.FloatTensor(classes.values)
            yield tokenizer.transform_texts(real_chunk.text.values), \
                  Variable(classes)
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


def load_settings_and_model(path: str, args=None) -> (Settings, 'WordRnn'):
    """ Load settings and model from the specified model path """
    from ptclf.models import WordRnn
    sqlite_con = sqlite3.connect(path + '.sqlite')
    settings = Settings.load(sqlite_con)
    if args:
        settings.add_args(args)
    model = WordRnn.load(settings)
    model.eval()
    return settings, model
