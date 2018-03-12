from ptclf.settings import parse_args
from ptclf.train import train


def test_basic_multiclass():
    train(parse_args(['train', '-i', 'assets/sentiment.csv', '-m', 'tmp', '--verbose', '0',
                      '--validate_path', 'assets/sentiment.csv']))
