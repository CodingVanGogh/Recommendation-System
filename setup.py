#!/usr/bin/env python

from setuptools import setup

setup(name='recommendation_system',
      version='1.0.0',
      description='Simple Recommendation System in Python3+ (Using Collaborative Filtering)',
      author='Coding Van Gogh',
      packages=['recommendation_system'],
      license="MIT",
      keywords="Recommendation System Movie recommender collaborative filtering low rank matrix factorization",
      url="https://github.com/CodingVanGogh/Recommendation-System",
      install_requires=['numpy>=1.13.1', 'pandas>=0.20.3', 'scipy>=0.19.1']
      )
