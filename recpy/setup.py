import numpy
from setuptools import Command, Extension, setup
from Cython.Build import cythonize

from Cython.Compiler.Options import _directive_defaults

_directive_defaults['linetrace'] = True
_directive_defaults['binding'] = True

extensions = [
    Extension(name='_cython._similarity',
              sources=["_cython/_similarity.pyx"], define_macros=[('CYTHON_TRACE', '1')])
]

setup(
    name='recpy',
    version="0.1.0",
    description='Recommender Systems framework for the 2016 Recsys Course at Polimi',
    url='https://github.com/mquad/recsys-course',
    author='Massimo Quadrana and Yashar Deldjoo',
    author_email='Massimo Quadrana',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Natural Language :: English',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Cython',
        'Operating System :: OS Independent',
        'Topic :: Software Development :: Libraries :: Python Modules',
        ],
    install_requires=['numpy', 'scipy>=0.16'],
    setup_requires=["Cython >= 0.19"],
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()]
)
