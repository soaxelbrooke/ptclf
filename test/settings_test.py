import sqlite3

from datetime import datetime
from ptclf.settings import Settings


def test_save_and_load_settings():
    settings = Settings({'id': 'test', 'classes': ('a', 'b'), 'created_at': datetime.utcnow()},
                        {'cuda': True, 'class_weights': (1.0, 2.0), 'batch_size': 10}, {})
    settings.try_defaults()
    con = sqlite3.connect(':memory:')
    settings.save(con)
    settings2 = Settings.load(con)
    assert settings.to_sql_insert() == settings2.to_sql_insert()
