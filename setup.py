from setuptools import setup, find_packages

__version__ = 'unknown'
for line in open('ReTiSAR/__init__.py'):
    if line.startswith('__version__'):
        exec(line)
        break

setup(
    name='ReTiSAR',
    description='Real-Time Spherical Array Renderer for binaural reproduction in Python',
    keywords='binauralaudio signal-processing microphone-array-processing python 3d-audio',
    version=__version__,

    url='https://github.com/AppliedAcousticsChalmers/pyBinauralTest',

    author='Hannes Helmholz',
    author_email='hannes.helmholz@chalmers.se',

    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Multimedia :: Sound/Audio',
        'Programming Language :: Python :: 3.7',
    ],

    python_requires='>=3.7',

    install_requires=[
        'jack-client >= 0.4.4',
        'matplotlib',
        'numpy >= 1.15.4',
        'pyfftw',
        'python-osc',
        'pyserial >= 3.4',
        'samplerate',
        'scipy',
        'soundfile >= 0.10.2',
        'sound_field_analysis > 0.3',
        # 'psutil',  # necessary when adjusting process priority
    ],
    # temporary fix until `sound_field_analysis > 0.3` is not available on PyPI
    # from index outside of PyPI ... did not work so far
    # `pip install . --extra-index-url https://github.com/AppliedAcousticsChalmers/sound_field_analysis-py/tarball/master#egg=sound_field_analysis`
    # from local directory in development mode (before installing this package)
    # `pip install -e ../sound_field_analysis-py/`
    # from git repository (before installing this package)
    # `pip install https://github.com/AppliedAcousticsChalmers/sound_field_analysis-py/tarball/master#egg=sound_field_analysis`

    extras_require={
        'benchmark': ['pandas', 'jinja2', 'matplotlib'],
    },

    package_data={
        '': ['res/*'],
    },

    packages=find_packages(),
)
