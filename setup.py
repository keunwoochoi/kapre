from setuptools import setup

setup(name='kapre',
      version='0.0.4',
      description='KAPRE: Keras Audio Preprocessors. A set of Keras layers for audio pre-processing easier deep learning',
      author='Keunwoo Choi',
      url='http://github.com/keunwoo/kapre/',
      download_url='http://github.com/keunwoochoi/kapre/releases',
      author_email='keunwoo.choi@qmul.ac.uk',
      license='MIT',
      packages=['kapre'],
      install_requires=[
        'keras >= 2.0.0',
        'numpy >= 1.8.0',
        'librosa >= 0.4',
        'pandas',
        'future'
      ],
      extras_require={
          'docs':['sphinx > 1.3', 'sphinx_rtd_theme'], # 'numpydoc', 
          'keras': ['keras'],
          },
      keywords='audio music deep learning keras',
      zip_safe=False)
