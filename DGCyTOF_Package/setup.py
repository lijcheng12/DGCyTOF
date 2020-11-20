#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ast
import io
import re
import os
from setuptools import find_packages, setup

DEPENDENCIES = ['scipy==1.5.2', 'numpy==1.19.1', 'seaborn==0.10.1', 'matplotlib==3.3.0', 'pandas==1.1.0', 'umap_learn==0.4.6', 'hdbscan==0.8.26', 'scikit_learn==0.23.2', 'scikit_plot==0.3.7', 'scikit_bio==0.5.6', 'umap-learn==0.4.6','torch==1.6.0','torchvision==0.7.0','tensorboard==2.3.0']
CURDIR = os.path.abspath(os.path.dirname(__file__))

with io.open(os.path.join(CURDIR, "README.md"), "r", encoding="utf-8") as f:
    README = f.read()

setup(
    name="DGCyTOF",
    version="1.0.0",
    author="Lijun Cheng, Pratik Karkhanis, Birkan Gokbag, and Lang Li",
    author_email="Lijun.Cheng@osumc.edu",
    description="DGCyTOF: Deep Learning with Graphical Cluster Visualization for CyTOF to Identify Cell Populations",
    long_description=README,
    url="https://github.com/lijcheng/DGCyTOF",
    package_dir={'DGCyTOF': 'DGCyTOF'},
    packages=['DGCyTOF'],
    include_package_data=True,
    keywords=[],
    scripts=[],
    zip_safe=False,
    install_requires=DEPENDENCIES,
    license="License :: OSI Approved :: GPL 3.0",
    classifiers=[
        "Programming Language :: Python",
        "Operating System :: OS Independent",
    ],
)