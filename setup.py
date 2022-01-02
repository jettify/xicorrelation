import os
import re

from setuptools import find_packages, setup

install_requires = ['numpy', 'scipy']


def _read(f):
    with open(os.path.join(os.path.dirname(__file__), f)) as f_:
        return f_.read().strip()


def _read_version():
    regexp = re.compile(r"^__version__\W*=\W*'([\d.abrc]+)'")
    init_py = os.path.join(
        os.path.dirname(__file__), 'xicorrelation', '__init__.py'
    )
    with open(init_py) as f:
        for line in f:
            match = regexp.match(line)
            if match is not None:
                return match.group(1)
        raise RuntimeError('Cannot find version in xicorrelation/__init__.py')


classifiers = [
    'License :: OSI Approved :: Apache Software License',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Operating System :: OS Independent',
    'Development Status :: 3 - Alpha',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
]

keywords = []

project_urls = {
    'Website': 'https://github.com/jettify/xicorrelation',
    'Documentation': 'https://xicorrelation.readthedocs.io',
    'Issues': 'https://github.com/jettify/xicorrelation/issues',
}


setup(
    name='torch-optimizer',
    version=_read_version(),
    description=('xicorrelation'),
    long_description='\n\n'.join((_read('README.rst'), _read('CHANGES.rst'))),
    long_description_content_type='text/x-rst',
    classifiers=classifiers,
    platforms=['POSIX'],
    author='Nikolay Novik',
    author_email='nickolainovik@gmail.com',
    url='https://github.com/jettify/xicorrelation',
    download_url='https://pypi.org/project/xicorrelation/',
    license='Apache 2',
    packages=find_packages(exclude=('tests',)),
    install_requires=install_requires,
    keywords=keywords,
    zip_safe=True,
    include_package_data=True,
    project_urls=project_urls,
    python_requires='>=3.6.0',
)
