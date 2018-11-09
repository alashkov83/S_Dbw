#!/usr/bin/python3
# coding=utf-8
from setuptools import setup

with open("README.md") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    requirements = f.readlines()

setup(
    name='s_dbw',
    version='0.3.0',
    python_requires='>=2.7',
    packages=['s_dbw',],
    url='https://github.com/alashkov83/S_Dbw',
    license='MIT License',
    platforms=['any'],
    author='Alexander Lashkov, Sergey Rubinsky, Polina Eistrikh-Heller',
    author_email='alashkov83@gmail.com',
    maintainer='Alexander Lashkov',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved',
        'Programming Language :: Python',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4'],
    description='Compute the S_Dbw validity index',
    long_description=long_description,
    keywords='clustering, cluster analysis, cluster validation',
    long_description_content_type="text/markdown",
    install_requires=requirements)
