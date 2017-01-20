from setuptools import setup
setup(name='kapre',
      version='0.0.2.3',
      description='KAPRE: Keras Audio Preprocessors. It provides keras layers for audio pre-processing for deep learning',
      url='http://github.com/keunwoo/kapre/',
      author='Keunwoo Choi',
      author_email='keunwoo.choi@qmul.ac.uk',
      license='MIT',
      packages=['kapre'],
      install_requires=[
        'keras >= 1.0.0', 
        'numpy >= 1.8.0',
        'librosa'
      ],
      download_url = 'https://github.com/keunwoo/kapre/tarball/0.0.1',
      keywords='audio music deep learning keras',
      zip_safe=False)
