from setuptools import setup
import platform
import sys

if sys.version_info < (3, 6):
    raise RuntimeError(
        "To use loft, Python version>=3.6 is highly recommended, you are using Python %s"
        % platform.python_version()
    )
else:
    pass

setup(name='loft',
    version='0.0.3',
    description='A fresh package for training & validating & evaling under data-parrallel distribution. This package is avaliable for torch and paddle deep-learning frameworks.',
    url='https://github.com/EagerSun/loft',
    author='Eager Sun',
    author_email='eagersyg@gmail.com',
    license='MIT',
    packages=['loft'],
    python_requires='>=3.6',
    install_requires=[
        'torch>=1.10.0',
        'paddlepaddle-gpu>=2.4.0',
        'tensorboard',
        'Pillow',
        'PyYAML',  
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    )
