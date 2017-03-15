from setuptools import setup

setup(name='kapre',
      version='0.0.3',
      description='KAPRE: Keras Audio Preprocessors. A set of Keras layers for audio pre-processing easier deep learning',
      url='http://github.com/keunwoo/kapre/',
      author='Keunwoo Choi',
      author_email='keunwoo.choi@qmul.ac.uk',
      license='MIT',
      packages=['kapre'],
      install_requires=[
        'keras >= 1.0.0', 
        'numpy >= 1.8.0',
        'librosa >= 0.4',
        'pandas',
      ],
      keywords='audio music deep learning keras',
      zip_safe=False)

#       download_url = 'https://github.com/keunwoo/kapre/tarball/0.0.3',
