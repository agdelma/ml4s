import setuptools

with open('README.md') as f:
    long_description = f.read()

setuptools.setup(
    name='ml4s',
    version='0.3.4',
    packages=setuptools.find_packages(),
    license='MIT',
    description='A python package implenting useful utilities for an introductory machine learning course taught at the University of Tennessee.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=['numpy','matplotlib','viznet'],
    python_requires='>=3.6',
    url='https://github.com/DelMaestroGroup/ml4s',
    author='Adrian Del Maestro',
    author_email='adrian@delmaestro.org',
    classifiers=[
   'License :: OSI Approved :: MIT License',
   'Programming Language :: Python :: 3.6',
   'Programming Language :: Python :: 3.7',
   'Topic :: Scientific/Engineering :: Physics']
)
