from setuptools import setup
setup(name='kapre',
      version='0.0.1',
      description='KAPRE: Keras Audio PREprocessing',
      url='http://github.com/keunwoo/kapre',
      author='Keunwoo Choi',
      author_email='gnuchoi@gmail.com',
      licsnse='MIT',
      packages=['kapre'],
      install_requires=[
        'keras', 'numpy', 'scipy',
      ],
      zip_safe=False)