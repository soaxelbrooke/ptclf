import logging

import os

from ptclf.hyperopt import hyperopt
from ptclf.predict import stdout_predict, predict
from ptclf.settings import parse_args
from ptclf.train import train
from ptclf.util import load_settings_and_model, convert_vecs_to_sqlite


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
    elif args.mode == 'convert-vecs':
        convert_vecs_to_sqlite(args.glove_path)


if __name__ == '__main__':
    main()
