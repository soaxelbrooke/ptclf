import json
import logging
import sqlite3
from subprocess import check_output
from uuid import uuid4

from ptclf.settings import Settings


def hyperopt(args):
    from bayes_opt import BayesianOptimization

    settings = Settings.from_args(args)
    with open(args.hyperopt_spec) as infile:
        spec = json.load(infile)

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