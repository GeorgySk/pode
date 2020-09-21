from pathlib import Path

from setuptools import (find_packages,
                        setup)

import pode

project_base_url = 'https://github.com/LostFan123/pode/'

setup(name=pode.__name__,
      packages=find_packages(exclude=('tests', 'tests.*')),
      version=pode.__version__,
      description=pode.__doc__,
      long_description=Path('README.md').read_text(encoding='utf-8'),
      long_description_content_type='text/markdown',
      author='Georgy Skorobogatov',
      author_email='georgy.skorobogatov@upc.edu',
      classifiers=[
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: Implementation :: CPython',
      ],
      license='MIT License',
      url=project_base_url,
      download_url=project_base_url + 'archive/master.zip',
      python_requires='>=3.8',
      install_requires=Path('requirements.txt').read_text(encoding='utf-8'))
