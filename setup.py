#/urs/bin/env python

__version__ = "0.0.1-dev"

from distutils.core import setup

classes = """
    Development Status :: 1 - Planning
    License :: OSI Approved :: BSD License
    Topic :: Software Development :: Libraries
    Topic :: Scientific/Engineering
    Topic :: Scientific/Engineering :: Statitics
    Programming Language :: Python
    Programming Language :: Python :: 3
    Programming Language :: Python :: 3.4
    Operating System :: Unix
    Operating System :: POSIX
    Operating System :: MacOS :: MacOS X
"""

setup(name='thesus',
      version="0.0.1-dev",
      license='BSD3',
      description="Code to support cluster simulation",
      # long_description=("Library for handling a metadata data dictionary "
      #                   "(and breaking the fourth wall)."),
      author="J W Debelius",
      author_email="justine.debelius@jhu.edu",
      # maintainer="J W Debelius",
      # maintainer_email="jdebelius@ucsd.edu",
      packages=['thesus', 'thesus.example_icons'],
      # install_requires=['numpy >= 1.10.0',
      #                   'pandas >= 0.23.4',
      #                   'nose >= 1.3.7',
      #                   ],
      )
