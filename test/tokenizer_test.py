import sqlite3

from ptclf.tokenizer import Tokenizer

TEST_TEXTS = ['''
    This is how we do it.  It's friday night, and I feel all right, and the party's here on the west
    side.
''']


def test_learn_vocab():
    """ We should be able to learn a tokenizer vocab from a simple string. """
    tok = Tokenizer(100)
    tok.fit_on_texts(TEST_TEXTS)
    seqs = tok.texts_to_sequences(TEST_TEXTS)
    assert len(seqs) == 1
    assert len(seqs[0]) == 28


def test_save_load_tokenizer():
    tok = Tokenizer(100)
    tok.fit_on_texts(TEST_TEXTS)
    con = sqlite3.connect(':memory:')
    tok.save(con)
    tok2 = Tokenizer.load(con)
    assert tok.texts_to_sequences(TEST_TEXTS) == tok2.texts_to_sequences(TEST_TEXTS)


def test_trim_after_fit():
    tok = Tokenizer(5, trim_after_fit=True)
    tok.fit_on_texts(TEST_TEXTS)
    assert len(tok.word_index) == 5
    assert len(tok.word_docs) == 0
    assert len(tok.word_counts) == 0


