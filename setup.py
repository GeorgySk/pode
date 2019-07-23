from pathlib import Path

from setuptools import (find_packages,
                        setup)

import pode

project_base_url = 'https://github.com/LostFan123/pode/'

setup_requires = [
    'pytest-runner>=4.2',
]
install_requires = Path('requirements.txt').read_text()
tests_require = Path('requirements-tests.txt').read_text()

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
      setup_requires=setup_requires,
      install_requires=install_requires,
      tests_require=tests_require)
