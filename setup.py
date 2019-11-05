from setuptools import setup

setup(name='kapre',
      version='0.1.5',
      description='Kapre: Keras Audio Preprocessors. Keras layers for audio pre-processing in deep learning',
      author='Keunwoo Choi',
      url='http://github.com/keunwoochoi/kapre/',
      # download_url='http://github.com/keunwoochoi/kapre/releases', # TODO
      author_email='keunwoo.choi@qmul.ac.uk',
      license='MIT',
      packages=['kapre'],
      install_requires=[
        'keras >= 2.0.0',
        'numpy >= 1.8.0',
        'librosa >= 0.5',
        'future'
      ],
      extras_require={
          'tests': ['tensorflow'],
       },
      keywords='audio music deep learning keras',
      zip_safe=False)
