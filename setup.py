from setuptools import setup

setup(
    name='kapre',
    version='0.2.0',
    description='Kapre: Keras Audio Preprocessors. Keras layers for audio pre-processing in deep learning',
    author='Keunwoo Choi',
    url='http://github.com/keunwoochoi/kapre/',
    author_email='gnuchoi@gmail.com',
    license='MIT',
    packages=['kapre'],
    package_data={
        'kapre': [
            'tests/fblog_8000_512.npy',
            'tests/speech_test_file.npz',
            'tests/test_audio_mel_g0.npy',
            'tests/test_audio_stft_g0.npy',
        ]
    },
    include_package_data=True,
    install_requires=[
        'numpy >= 1.8.0',
        'librosa >= 0.7.2',
        'tensorflow >= 1.15',
    ],
    keywords='audio music deep learning keras',
    zip_safe=False,
)
