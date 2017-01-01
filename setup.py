from setuptools import setup
setup(name='kapre',
      version='0.0.1',
      description='KAPRE: Keras Audio PREprocessors',
      url='http://github.com/keunwoo/kapre/',
      author='Keunwoo Choi',
      author_email='keunwoo.choi@qmul.ac.uk',
      license='MIT',
      packages=['kapre'],
      install_requires=[
        'keras >= 1.0.0', 
        'numpy >= 1.8.0',
      ],
      keywords='audio music deep learning keras',
      zip_safe=False)
