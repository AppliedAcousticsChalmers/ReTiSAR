from setuptools import find_packages, setup

__version__ = 'unknown'
for line in open('ReTiSAR/__init__.py'):
    if line.startswith('__version__'):
        exec(line)
        break

setup(
    name='ReTiSAR',
    description='Real-Time Spherical Array Renderer for binaural reproduction in Python',
    keywords='binauralaudio signal-processing microphone-array-processing python 3d-audio',
    url='https://github.com/AppliedAcousticsChalmers/ReTiSAR',
    version=__version__,

    author='Hannes Helmholz',
    author_email='hannes.helmholz@chalmers.se',

    long_description=open('README.md', mode='r', encoding='utf-8').read(),
    long_description_content_type='text/markdown',

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'License :: Other/Proprietary License',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Multimedia :: Sound/Audio',
    ],

    python_requires='>=3.7',

    install_requires=[
        'jack-client >= 0.4.4',
        'matplotlib',
        'numpy >= 1.15.4',
        'pyfftw',
        'python-osc',
        'pyserial >= 3.4',
        'pysofaconventions >= 0.1.5',
        'samplerate',
        'scipy',
        'soundfile >= 0.10.2',
        'sound_field_analysis >= 2019.8.15',
        # 'psutil',  # necessary when adjusting process priority
    ],

    extras_require={
        'benchmark': [
            'jinja2',
            'natsort',
            'pandas',
        ],
    },

    package_data={
        '': ['res/*'],
    },

    packages=find_packages(),
)
