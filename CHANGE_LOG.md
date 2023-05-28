# Change Log

* __unreleased__
  * Update `conda` environment setup with latest version of `sound-field-analysis` as long as the PyPI package is not yet available
  * Update of validation script for improved readability
  * Improve startup behaviour of the SH renderer for _JACK_ configurations without physical input ports
  * Update of command line argument parser to be more forgiving with source position strings
  * Improve `JackPlayer` to enforce providing C-contiguous data to _JACK_
  * Consolidation of `Python 3.11` and `numpy>=1.24.0` compatibility (requires `sound-field-analysis>=2022.12.29`)
  * Move change log information from README to separate CHANGE_LOG file
  * Change default FFTW effort to "FFTW_MEASURE" (lowered from "FFTW_PATIENT")

* __v2021.07.26__ ([__v2021.TASLP__](https://github.com/AppliedAcousticsChalmers/ReTiSAR/releases/tag/v2021.TASLP))
  * Addition of references to data set for _IEEE/ACM Transactions on Audio, Speech, and Language Processing_ [[14]](README.md#references) publication
  * Improve logging of loaded FFTW wisdom
  * Improve behaviour of the SH renderer to create input ports even if less physical recording ports are present
  * Update `DataRetriever` to be more resilient against faulty downloads
* __v2021.03.30__
  * Addition of Zylia _ZM-1_ array example recording and live-stream configurations
* __v2020.11.23__
  * Consolidation of `Python 3.9` compatibility
  * Consolidation of _Linux_ compatibility (no modifications were required; tested with _Jack_ `1.9.16` on kernel `5.9.1-1-rt19-MANJARO`)
* __v2020.10.21__
  * Improvement of establishing _JACK_ and/or OS specific client name length limitation of `JackClient`
* __v2020.10.15__ ([__v2020.FA__](https://github.com/AppliedAcousticsChalmers/ReTiSAR/releases/tag/v2020.FA))
  * Addition of references to data set for _Forum Acusticum_ [[10]](README.md#references) publication
* __v2020.9.10__
  * Enforcement of `Black >= 20.8b1` code style
* __v2020.8.20__
  * Extension of `JackPlayer` to make autoplay behaviour configurable (via `config` parameter or command line argument)
* __v2020.8.13__
  * Update of OSC Remote Demo to reset (after not receiving data) and clip displayed RMS values
  * Improvement of FFTW wisdom verification to be more error proof
* __v2020.7.16__
  * Addition of experimental `SECTORIAL_DEGREE_SELECTION` and `EQUATORIAL_DEGREE_SELECTION` SH weighting techniques (partial elimination of HRIR elevation queues)
  * Update of plotting during and after application of SH compensation techniques
* __v2020.7.7__
  * Addition of HRIR and HPCF source files to _SADIE II_ database [[8]](README.md#references)
  * Extension of `DataRetriever` to automatically extract requested resources from downloaded `*.zip` archives
* __v2020.7.4__
  * Introduction of FFTW wisdom file signature verification (in order to update any already accumulated wisdom run with `--PYFFTW_LEGACY_FILE=log/pyfftw_wisdom.bin` once)
  * Fixes for further SonarLint security and code style recommendations
* __v2020.7.1__
  * Update and addition of further _WDR Cologne_ ARIR source files (linking to Zenodo data set)
  * Hack for Modal Radial Filters generation in open / cardioid SMA configurations (unfortunately this metadata is not directly available in the SOFA ARIR files)
* __v2020.4.8__
  * Improvement of IIR pink noise generation (continuous utilization of internal filter delay conditions)
  * Improvement of IIR pink noise generation (employment of SOS instead of BA coefficients)
  * Addition of IIR _Eigenmike_ coloration noise generation according to [[10]](README.md#references)
* __v2020.4.3__
  * Improvement of white noise generation (vastly improved performance due to `numpy SFC64` generator)
  * Enabling of `JackGenerator` (and derivatives) to operate in single precision for improved performance
* __v2020.3.3__
  * Addition of further simulated array data sets
* __v2020.2.24__
  * Consolidation of `Python 3.8` compatibility
  * Introduction of `multiprocessing` context for compatibility
  * Enforcement of `Black` code style
* __v2020.2.14__
  * Addition of TH Cologne _HØSMA-7N_ array configuration
* __v2020.2.10__
  * Addition of project community information (contributing, code of conduct, issue templates)
* __v2020.2.7__
  * Extension of `DataRetriever` to automatically download data files
  * Addition of missing ignored project resources
* __v2020.2.2__
  * Change of default rendering configuration to contained _Eigenmike_ recording
  * Update of README structure (including Quickstart section)
* __v2020.1.30__
  * First publication of code

* Pre-release ([__v2020.ICASSP__](https://github.com/AppliedAcousticsChalmers/ReTiSAR/releases/tag/v2020.ICASSP))
  * Contains the older original code state for the _ICASSP_ [[13]](README.md#references) publication
* Pre-release ([__v2019.DAGA__](https://github.com/AppliedAcousticsChalmers/ReTiSAR/releases/tag/v2019.DAGA))
  * Contains the older original code state for the initial _DAGA_ [[1]](README.md#references) publication
