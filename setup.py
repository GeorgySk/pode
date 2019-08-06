from pathlib import Path

from setuptools import (find_packages,
                        setup)

import pode

project_base_url = 'https://github.com/LostFan123/pode/'

install_requires = [
    'lz>=0.8.1',
    'networkx>=2.3',
    'shapely>=1.6.4.post2'
]
setup_requires = [
    'pytest-runner>=4.2',
]
tests_require = [
    'pytest>=3.8.1',
    'pytest-cov>=2.6.0',
    'hypothesis>=3.73.1',
]

setup(name='pode',
      packages=find_packages(exclude=('tests', 'tests.*')),
      version=pode.__version__,
      description=pode.__doc__,
      long_description=Path('README.md').read_text(encoding='utf-8'),
      long_description_content_type='text/markdown',
      author='Georgy Skorobogatov',
      author_email='skorobogatov@phystech.edu',
      url=project_base_url,
      download_url=project_base_url + 'archive/master.zip',
      python_requires='>=3.6',
      install_requires=install_requires,
      setup_requires=setup_requires,
      tests_require=tests_require)
