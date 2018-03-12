""" Utility class for tracking experiments in an sqlite database """
import os
import sqlite3
import time
from uuid import uuid4


class SqliteExperiment:
    def __init__(self, hparams, metrics, experiment_id=None):
        self.experiment_id = experiment_id or str(uuid4())
        self.hparams = hparams
        self.metrics = metrics
        self.metric_names = ['experiment_id', 'measured_at'] + [n for n, t in metrics]
        self.log_every = int(os.environ.get('LOG_EVERY', 10000))
        self.last_log = None
        self.last_epoch = None
        self.db = sqlite3.connect('experiments.sqlite')
        self.ensure_tables()

    def ensure_tables(self):
        """ Create tables for metrics and hyper params if they don't exist """
        self.db.execute('''
CREATE TABLE IF NOT EXISTS hparams (
  experiment_id text primary key,
  {}
)
        '''.format(self.to_sql_column_defs(self.hparams).strip(',')))

        self.db.execute('''
CREATE TABLE IF NOT EXISTS metrics (
  experiment_id text,
  measured_at int,
  {}
)
        '''.format(self.to_sql_column_defs(self.metrics).strip(',')))
        self.db.commit()

    @classmethod
    def to_sqlite_col_type(cls, col_type):
        return {
            int: 'integer',
            float: 'real',
            str: 'text',
            bool: 'integer',
        }[col_type]

    def to_sql_column_defs(self, spec):
        return ',\n'.join([
            '{} {}'.format(col_name, self.to_sqlite_col_type(col_type))
            for col_name, col_type in spec
        ]) + ','

    def log_hparams(self, hparams):
        hparam_values = [self.experiment_id] + [hparams[name] for name, _type in self.hparams]
        for idx, hparam in enumerate(hparam_values):
            if isinstance(hparam, list):
                hparam_values[idx] = ','.join(map(str, hparam))
        self.db.execute('''
insert into hparams values ({})
        '''.format(', '.join(['?'] * len(hparam_values))), hparam_values)
        self.db.commit()

    def should_log(self, epoch, step):
        should_log = False
        if (self.last_epoch is None or self.last_log is None) \
                or (self.last_epoch < epoch) \
                or ((self.last_log + self.log_every) < step):
            self.last_epoch = epoch
            self.last_log = step
            should_log = True
        return should_log

    def log_metrics(self, epoch, step, metrics, force=False):
        if not force and not self.should_log(epoch, step):
            return
        metric_values = [self.experiment_id, time.time()] + \
                        [metrics.get(name) for name, _type in self.metrics]
        self.db.execute(
            '''
insert into metrics ({}) values ({})
            '''.format(', '.join(self.metric_names), ', '.join(['?'] * len(metric_values))),
            metric_values)
        self.db.commit()

