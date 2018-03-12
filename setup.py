# coding=utf-8

from setuptools import setup

with open('requirements.txt') as infile:
    dependencies = [line.strip().split('--')[0] for line in infile if len(line) > 0]

setup(name='ptclf',
      version='0.2.0',
      description='PyTorch Text Classifier',
      url='https://github.com/soaxelbrooke/ptclf',
      author='Stuart Axelbrooke',
      author_email='stuart@axelbrooke.com',
      packages=['ptclf'],
      setup_requires=['pytest-runner'],
      tests_require=['pytest', 'hypothesis'],
      install_requires=dependencies,
      zip_safe=False)