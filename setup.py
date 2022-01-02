import os

from setuptools import find_packages, setup

install_requires = ['numpy', 'scipy']


def _read(f):
    with open(os.path.join(os.path.dirname(__file__), f)) as f_:
        return f_.read().strip()


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

use_scm_version = {
    'write_to': '_version.py',
    'write_to_template': '__version__ = "{version}"',
}


setup(
    name='torch-optimizer',
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
    setup_requires=[
        "setuptools>=45",
        "setuptools_scm",
        "setuptools_scm_git_archive",
        "wheel",
    ],
    keywords=keywords,
    zip_safe=True,
    include_package_data=True,
    project_urls=project_urls,
    python_requires='>=3.6.0',
    use_scm_version=True,
)
