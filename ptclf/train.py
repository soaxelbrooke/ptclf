import logging
import os
import sqlite3

COMET_API_KEY = os.environ.get('COMET_API_KEY')
COMET_PROJECT = os.environ.get('COMET_PROJECT')
COMET_LOG_CODE = os.environ.get('COMET_LOG_CODE', '').lower() == 'true'

if COMET_API_KEY:
    from comet_ml import Experiment

from collections import deque
from datetime import datetime
from subprocess import check_output
from unittest.mock import MagicMock

import numpy
import torch
from torch import nn

from ptclf.experiment import SqliteExperiment
from ptclf.models import WordRnn, build_model
from ptclf.settings import Settings
from ptclf.tokenizer import get_tokenizer
from ptclf.util import get_classes, count_classes, progress, train_batch_iter, dev_batch_iter, \
    num_correct, auroc


def train(args):
    """ Trains RNN based on provided arguments """
    if args.random_seed:
        torch.manual_seed(args.random_seed)
        numpy.random.seed(args.random_seed)

    settings = Settings.from_args(args)
    settings.save(sqlite3.connect(settings.model_path + '.sqlite'))
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
        logging.debug('Model loaded from {}, continuing training'.format(settings.model_path))
    tokenizer = get_tokenizer(settings)

    if settings.class_weights:
        logging.debug('Class weights specified: {}'.format(settings.class_weights))
        class_weights = torch.FloatTensor(settings.class_weights)
    elif settings.learn_class_weights:
        logging.debug('Learning class weights...')
        class_counts = count_classes(settings)
        class_weights = torch.FloatTensor([sum(class_counts.values()) / class_counts[c]
                                           for c in get_classes(settings)])
        class_weights /= class_weights.min()
        assert sum(class_counts.values()) > 0, \
            "Didn't find any examples of any classes (input iterator was empty)"
        logging.debug('Inferred class weights: {}'.format(class_weights))
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
        logging.debug('Beginning training.')
        for epoch in range(settings.epochs):
            model.train()
            train_epoch(settings, model, criterion, optimizer, epoch, comet_experiment,
                        train_batches, sle)
            if settings.get('validate_path'):
                model.eval()
                val_loss = score_model(settings, model, criterion, epoch, comet_experiment,
                                       dev_batches, sle)
                scheduler.step(val_loss, epoch=epoch)
            if settings.cuda:
                model.switch_to_cpu()
            torch.save(model.state_dict(), settings.model_path + '.bin')
            if settings.cuda:
                model.switch_to_gpu()
            logging.debug('Model saved at {}'.format(settings.model_path))
            if settings.epoch_shell_callback:
                logging.debug('Executing epoch callback: {}'.format(settings.epoch_shell_callback))
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

    bar = progress(settings, desc='Epoch {}'.format(epoch), total=tokenizer.document_count)

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
        bar.set_postfix(loss=rolling_loss, acc=accuracy)
        bar.update(batch_x.size(1))

        comet_experiment.log_step(step)
        comet_experiment.log_loss(rolling_loss)
        comet_experiment.log_accuracy(accuracy)
        epoch_period = (datetime.now() - started).total_seconds()
        comet_experiment.log_metric('train_items_per_second', seen / epoch_period)

    bar.close()
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

    return output, float(loss.item())


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
        losses.append(loss.item())
        seen += batch_x.size(1)
        if settings.loss_fn != 'CrossEntropy' and settings.loss_fn != 'NLL':
            accuracies.append(auroc(output, batch_y))
        else:
            accuracies.append(num_correct(output, batch_y) / batch_x.shape[1])

    accuracy = float(numpy.mean(accuracies))
    period = (datetime.now() - started).total_seconds()
    mean_loss = float(sum(losses) / len(losses))
    sle.log_metrics(epoch, seen, {'dev_loss': mean_loss, 'epoch': epoch,
                                  'dev_acc': accuracy, 'score_per_second': seen / period},
                    force=True)
    comet_experiment.log_metric('dev_loss', mean_loss)
    comet_experiment.log_metric('dev_acc', accuracy)
    if settings.verbose > 0:
        logging.info('Epoch: {}\t  Dev accuracy: {:.3f}\t  Dev loss: {:.3f}, Scored/sec: {}'.format(
            epoch, accuracy, sum(losses) / len(losses), seen / period))
    return sum(losses) / len(losses)
