import logging
from datetime import datetime

import pandas
import toolz

from ptclf.tokenizer import get_tokenizer
from ptclf.util import load_settings_and_model


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
