import os
from hypothesis import given, settings
from hypothesis import strategies as s

from ptclf.settings import parse_args
from ptclf.train import train

arg_bool = s.one_of([s.just('true'), s.just('false')])


def test_basic_multiclass():
    train(parse_args(['train', '-i', 'assets/sentiment.csv', '-m', 'tmp', '--verbose', '0',
                      '--validate_path', 'assets/sentiment.csv']))


@settings(max_examples=int(os.environ.get('RNN_TEST_DRAWS', 1)),
          deadline=int(os.environ.get('RNN_TEST_DEADLINE_MS', 1000)))
@given(
    s.one_of([s.just(v) for v in ('lstm', 'gru')]),
    s.one_of([s.just(v) for v in ('last', 'max', 'maxavg', 'attention')]),
    s.booleans(),
    s.booleans(),
    s.one_of([s.just(v) for v in ('NLL', 'CrossEntropy')]),
)
def test_train_multiclass(rnn_kind, context_mode, learn_rnn_init, bidirectional, loss_fn):
    args = [
        'train', '-i', 'assets/sentiment.csv', '-m', 'tmp', '--verbose', '0', '-c', '5', '-e', '10',
        '--validate_path', 'assets/sentiment.csv',
        '--rnn', rnn_kind,
        '--context_mode', context_mode,
        '--loss_fn', loss_fn,
    ]

    if learn_rnn_init:
        args.append('--learn_rnn_init')

    if bidirectional:
        args.append('--bidirectional')

    train(parse_args(args))
