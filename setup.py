from setuptools import setup

setup(name='kapre',
      version='0.1.6',
      description='Kapre: Keras Audio Preprocessors. Keras layers for audio pre-processing in deep learning',
      author='Keunwoo Choi',
      url='http://github.com/keunwoochoi/kapre/',
      # download_url='http://github.com/keunwoochoi/kapre/releases', # TODO
      author_email='keunwoo.choi@qmul.ac.uk',
      license='MIT',
      packages=['kapre'],
      package_data={'': ['tests/speech_test_file.npz', 'tests/test_audio_mel_g0.npy', 'tests/test_audio_stft_g0.npy']},
      include_package_data=True,
      install_requires=[
        'numpy >= 1.8.0',
        'librosa >= 0.5',
        'tensorflow >= 1.15',
        'future'
      ],
      extras_require={
          'tests': ['tensorflow'],
       },
      keywords='audio music deep learning keras',
      zip_safe=False)
