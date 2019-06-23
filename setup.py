from setuptools import setup

setup(name='SeqWORDS',
      version='0.1',
      description='This is a Chinese words segmentation method.',
      author='Harry Wu',
      author_email='tf00604520@gmail.com',
      license='MIT',
      url='https://github.com/kkuanhui/SeqWORDS',
      packages=['SeqWORDS'],
      install_requires=[
          'numpy',
      ],
      zip_safe=False)
