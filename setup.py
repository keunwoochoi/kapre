from setuptools import setup

setup(
    name='kapre',
    version='0.3.5',
    description='Kapre: Keras Audio Preprocessors. Tensorflow.Keras layers for audio pre-processing in deep learning',
    author='Keunwoo Choi',
    url='http://github.com/keunwoochoi/kapre/',
    author_email='gnuchoi@gmail.com',
    license='MIT',
    package_dir = {
            'kapre': 'kapre',
            'kapre.tflite': 'kapre/tflite',
            },
    packages=['kapre','kapre.tflite'],
    package_data={
        'kapre': [
            'tests/speech_test_file.npz',
        ]
    },
    include_package_data=True,
    install_requires=[
        'numpy >= 1.18.5',
        'librosa >= 0.7.2',
        'tensorflow>=2.0.0'
    ],
    keywords='audio music speech sound deep learning keras tensorflow',
    zip_safe=False,
)
