from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='torchstruct',
    version='0.1.1',
    author='Tomasz Wrona',
    author_email='tomasz@wrona.me',
    description='',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/iamhatesz/torchstruct',
    packages=find_packages(),
    py_modules=['torchstruct'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Version Control :: Git',
        'Topic :: Utilities',
        'Typing :: Typed'
    ],
    python_requires='>=3',
    install_requires=['torch'],
    tests_require=['pytest']
)
