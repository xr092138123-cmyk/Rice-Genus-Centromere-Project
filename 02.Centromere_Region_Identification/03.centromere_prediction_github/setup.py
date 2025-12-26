#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup script for Centromere Area Prediction
"""

from setuptools import setup, find_packages
import os

# Read README
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

# Read requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    requirements = []
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    requirements.append(line)
    return requirements

setup(
    name='centromere-prediction',
    version='1.0.0',
    description='Transformer-based centromere area prediction model',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    author='Centromere Prediction Team',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/centromere_prediction',
    packages=find_packages(),
    install_requires=read_requirements(),
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='centromere genomics deep-learning transformer bioinformatics',
    project_urls={
        'Bug Reports': 'https://github.com/yourusername/centromere_prediction/issues',
        'Source': 'https://github.com/yourusername/centromere_prediction',
        'Documentation': 'https://github.com/yourusername/centromere_prediction/blob/main/README.md',
    },
    entry_points={
        'console_scripts': [
            'centromere-train=src.training.train:main',
            'centromere-predict=src.training.inference:main',
        ],
    },
)


