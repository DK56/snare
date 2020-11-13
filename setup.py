import io
import os
from setuptools import setup, find_namespace_packages

dir_path = os.path.abspath(os.path.dirname(__file__))
with io.open(os.path.join(dir_path, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='snare-ml',
    version='0.1.0',
    description='Scorebased Neural Architecture REduction',
    long_description=long_description,
    author='Tobias Viehmann',
    author_email='tobias.viehmann@mail.de',
    python_requires='>=3.6.0',
    url='https://github.com/dk56/snare',
    packages=find_namespace_packages(include=['snare.*']),
    install_requires=['numpy>=1.16.0',
                      'tensorflow-gpu==2.3.1'],
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Development Status :: 4 - Beta',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
    ]
)
