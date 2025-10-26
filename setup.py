import re
import os
from setuptools import setup

def get_version():
    with open(os.path.join('kapre', '__init__.py'), 'r') as f:
        return re.search(r"__version__ = ['\"]([^'\"]+)['\"]", f.read()).group(1)

setup(
    name='kapre',
    version=get_version(),
    description='Kapre: Keras Audio Preprocessors. Tensorflow.Keras layers for audio pre-processing in deep learning',
    author='Keunwoo Choi',
    url='http://github.com/keunwoochoi/kapre/',
    author_email='gnuchoi@gmail.com',
    license='MIT',
    packages=['kapre'],
    package_data={
        'kapre': [
            'tests/speech_test_file.npz',
        ]
    },
    include_package_data=True,
    python_requires='>=3.8',
    install_requires=[
        'numpy >= 2.0.0',
        'librosa >= 0.11.0, < 1.0.0',
        'tensorflow >= 2.16.0, < 2.21.0',
    ],
    keywords='audio music speech sound deep learning keras tensorflow',
    zip_safe=False,
)
