#/urs/bin/env python

__version__ = "0.0.1"

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

setup(name='theseus',
      version="0.0.1",
      license='BSD-3',
      description="Multi-cohort clustering simulation",
      long_description=("Support code for multi-cohort simulations for Kress et al"),
      author="J W Debelius",
      author_email="justine.debelius@jhu.edu",
      maintainer="J W Debelius",
      maintainer_email="justine.debelius@jhu.edu",
      packages=['theseus', 
                'theseus.tests',
                ]
      # install_requires=['numpy >= 1.10.0',
                        # 'pandas >= 0.23.4',
                        # 'nose >= 1.3.7',
                        # ],
      )
