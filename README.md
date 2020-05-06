# ReTiSAR
Implementation of the Real-Time Spherical Microphone Renderer for binaural reproduction in _Python_ [[1]](#references).

## Requirements:
* _MacOS_ (other OS not tested)
* [_JACK_ library](http://jackaudio.org/downloads/) (on _MacOS_ the prebuilt binary works best)
* [_Conda_ installation](https://conda.io/en/master/miniconda.html) (`miniconda` is enough, highly recommended to get [Intel _MKL_](https://software.intel.com/en-us/articles/using-intel-distribution-for-python-with-anaconda) or alternatively [_OpenBLAS_](https://github.com/conda-forge/openblas-feedstock) optimized `numpy` version, automatically done in [setup section](#setup))
* [_Python_ installation](https://www.python.org/downloads/), recommended way is to use _Conda_ (automatically done in [setup section](#setup))
* Installation of the required _Python_ packages (see [setup section](#setup))
* __Optional:__ [_OSC_ client](http://opensoundcontrol.org/implementations) (see [remote control section](#remote-control))

## Checkout / Download:
* Clone _git_ repository via command line<br/>
`git clone https://github.com/AppliedAcousticsChalmers/ReTiSAR.git`
  * __Alternative 1:__ Clone via any other _git_ client<br/>
  _GitHub desktop client_ or _PyCharm_ worked well after login, since login credentials on _MacOS_ command line did not work (maybe because of two-factor-authentication?)
  * __Alternative 2:__ Download and extract manually from provided URL

## Setup:
* Navigate into repository (the directory containing _setup.py_)<br/>
`cd ReTiSAR/`
* Make sure that _Conda_ is up to date<br/>
`conda update conda`
* __Optional:__ Delete existing _Conda_ environment<br/>
`conda env remove --name ReTiSAR`
* Create new _Conda_ environment from the specified requirements<br/>
`conda env create --file environment.yml`
* Activate _Conda_ environment<br/>
`source activate ReTiSAR`

## Execution:
* __Optional:__ In case you have never started the _JACK_ audio server on your system or want to make sure it initializes with appropriate values. Open the _JackPilot_ application set your system specific default settings.<br/>
At this point the only relevant _JACK_ audio server setting is the sampling frequency, which has to match the sampling frequency of your rendered audio source file or stream (no resampling will be applied for that specific file).<br/>
__!!__ All here provided examples are supposed to be run at a sampling frequency of __48 kHz__ __!!__
* Run package with default parameters<br/>
`python -m ReTiSAR`<br/>
__Remark:__ In case the rendering takes very long to start (after the message _"initializing FFTW DFT optimization ..."_), you might want to endure this long computation time once (per rendering configuration) or lower your [FFTW](http://www.fftw.org/fftw3_doc/Planner-Flags.html) planner effort (see `--help`)
  * __Alternative 1:__ Modify configuration by changing default parameters in [config.py](ReTiSAR/config.py) (prepared block comments for the specific configurations below exist)
  * __Alternative 2:__ Modify configuration by command line arguments (like in the following examples), here showing possible parameters<br/>
  `python -m ReTiSAR --help`
* __Rendering performance --__ follow these remarks to expect continuous and artifact free rendering:<br/>
  * Optional components like array pre-rendering, headphone equalization, noise generation, etc. will save performance in case they are not deployed
  * Extended IR lengths (particularly array pre-rendering) will massively increase the computational load depending on the chosen block length (partitioned convolution)
  * Currently there is no partitioned convolution for the main binaural renderer with SH based processing, hence the FIR taps of applied HRIR, Modal Radial Filters and further compensations (e.g. Spherical Head Filter) need to cumulatively fit inside the chosen block length
  * Higher block length means lower computational load in real-time rendering but also increased system latency (relevant for head-tracking and array live-rendering)
  * Adjust output levels of all rendering components (default parameters chosen accordingly) to prevent signal clipping (indicated by warnings during execution)
  * Check _JACK_ system load (i.e. _JackPilot_) to be below approx. 95% to prevent dropouts (OS reported system load is not a good indicator)
  * Check _JACK_ detected dropouts ("xruns" indicated during execution)
  * Most of all, use your ears! If something sounds wrong, there probably is something going wrong...

The following parameters are all optional and available in combinations with the named configurations subsequently:<br/>
* Run package with possibly lower block size for decreased latency (choose value according to individual rendering modes and performance of your system)<br/>
`python -m ReTiSAR -b=512`
* Run package with processing in 64 bit double precision for better accuracy (default is __32 bit__ single precision for
 better performance)<br/>
`python -m ReTiSAR -SP=FALSE`
* Run package without truncation of any IR set after load (default level is __-60 dB__ relative under peak to save performance)<br/>
`python -m ReTiSAR -irt=0`
* Run package with individual head-tracking device (paths system dependent)<br/>
`python -m ReTiSAR -tt=AUTO_ROTATE` (which is the default behaviour) or<br/>
`python -m ReTiSAR -tt=RAZOR_AHRS -t=/dev/tty.usbserial-AH03F9XC` or<br/>
`python -m ReTiSAR -tt=POLHEMUS_PATRIOT -t=/dev/tty.UC-232AC`
* Run package with individual headphone equalization (partitioned convolution)<br/>
`python -m ReTiSAR -hp=res/HpIR/KU100_THK/AKG-K702.wav` (which is the default behaviour) or<br/>
`python -m ReTiSAR -hp=res/HpIR/KU100_THK/Sennheiser-HD650.wav` or<br/>
`python -m ReTiSAR -hp=res/HpIR/FABIAN_TUB/Sennheiser-HD650.wav`
* Run package with individual spherical harmonics processing compensation techniques<br/>
  * __Modal Radial Filters__ (always applied) with individual amplification soft-limiting in dB according to [[2]](#references)<br/>
  `python -m ReTiSAR -arr=18`
  * __Spherical Head Filter__ according to [[3]](#references)<br/>
  `python -m ReTiSAR -sht=SHF`<br/>
  * __Spherical Harmonics Tapering__ in combination with __Spherical Head Filter__ according to [[4]](#references)<br/>
  `python -m ReTiSAR -sht=SHT+SHF`<br/>

Specific configurations:<br/>
* HRTF dataset of artificial head _Neumann KU100_ from [[5]](#references) as _SOFA_ files according to [[6]](#references)

* Run package as array stream renderer of _Eigenmike_ (**Oc**ulus or **Ch**almers)<br/>
`python -m ReTiSAR -sh=4 -tt=NONE -s=None -ar=res/ARIR/Eigenmike_OC_calibration_struct.mat -art=AS_MIRO -arl=0 -hr=res/HRIR/KU100_THK/L2702.sofa -hrt=HRIR_SOFA` or<br/>
`python -m ReTiSAR -sh=4 -tt=NONE -s=None -ar=res/ARIR/Eigenmike_CH_calibration_struct.mat -art=AS_MIRO -arl=0 -hr=res/HRIR/KU100_THK/L2702.sofa -hrt=HRIR_SOFA`
* Run package as array recording renderer of _Eigenmike_<br/>
`python -m ReTiSAR -sh=4 -tt=NONE -s=res/source/Eigenmike_CH_LL_around.wav -ar=res/ARIR/Eigenmike_CH_calibration_struct.mat -art=AS_MIRO -arl=0 -hr=res/HRIR/KU100_THK/L2702.sofa -hrt=HRIR_SOFA` or<br/>
`python -m ReTiSAR -sh=4 -tt=NONE -s=res/source/Eigenmike_CH_LL_updown.wav -ar=res/ARIR/Eigenmike_CH_calibration_struct.mat -art=AS_MIRO -arl=0 -hr=res/HRIR/KU100_THK/L2702.sofa -hrt=HRIR_SOFA`

* Run package as array IR renderer of _Eigenmike_<br/>
`python -m ReTiSAR -sh=4 -s=res/source/Drums_48.wav -ar=res/ARIR/Eigenmike_synthetic_struct.mat -art=ARIR_MIRO -arl=0 -hr=res/HRIR/KU100_THK/L2702.sofa -hrt=HRIR_SOFA` or<br/>
`python -m ReTiSAR -sh=4 -s=res/source/Drums_48.wav -ar=res/ARIR/Eigenmike_CH_anechoic_struct.mat -art=ARIR_MIRO -arl=0 -hr=res/HRIR/KU100_THK/L2702.sofa -hrt=HRIR_SOFA`
* Run package as array IR renderer with 50 channels corresponding to 5th order from [[7]](#references)<br/>
`python -m ReTiSAR -sh=5 -s=res/source/Drums_48.wav -sp="[(-37,0)]" -ar=res/ARIR/CR1_VSA_50RS_L.sofa -art=ARIR_SOFA -arl=-12 -hr=res/HRIR/KU100_THK/L2702.sofa -hrt=HRIR_SOFA` or<br/>
`python -m ReTiSAR -sh=5 -s=res/source/Drums_48.wav -ar=res/ARIR/LBS_VSA_50RS_PAC.sofa -art=ARIR_SOFA -arl=-12 -hr=res/HRIR/KU100_THK/L2702.sofa -hrt=HRIR_SOFA`
* Run package as array IR renderer with 110 channels corresponding to 8th order from [[7]](#references)<br/>
`python -m ReTiSAR -sh=8 -s=res/source/Drums_48.wav -sp="[(-37,0)]" -ar=res/ARIR/CR1_VSA_110RS_L.sofa -art=ARIR_SOFA -arl=-12 -hr=res/HRIR/KU100_THK/L2702.sofa -hrt=HRIR_SOFA` or<br/>
`python -m ReTiSAR -sh=8 -s=res/source/Drums_48.wav -ar=res/ARIR/LBS_VSA_110RS_PAC.sofa -art=ARIR_SOFA -arl=-12 -hr=res/HRIR/KU100_THK/L2702.sofa -hrt=HRIR_SOFA`

* Run package as BRIR renderer with partitioned convolution in frequency domain from [[8]](#references)<br/>
`python -m ReTiSAR -s=res/source/Drums_48.wav -hr=res/BRIR/CR1_VSA_110RS_L_SSR_SFA_-37_SOFA_RFI.wav -hrt=BRIR_SSR`
* Run package as "binauralizer" (rendering of virtual sound sources over HRTF) with partitioned convolution in frequency domain (sampling frequency at __44100 Hz__ in this case)<br/>
`python -m ReTiSAR -s=res/source/PinkMartini_Lilly.wav -sp="[(30, 0),(-30, 0)]" -hr=res/HRIR/FABIAN_TUB/hrirs_fabian.wav -hrt=HRIR_SSR`

## Remote Control:
* Certain parameters of the running real-time application can be remote controlled via _Open Sound Control_. Individual clients can be accessed by targeting them with specific _OSC_ commands.<br/>
Depending on the current configuration and rendering mode different commands will be available, i.e. arbitrary combinations of the following targets and values:<br/>
`/generator/volume 0`, `/generator/volume -12`<br/>
`/prerenderer/mute 1`, `/prerenderer/mute 0`, `/prerenderer/mute -1`, `/prerenderer/mute`<br/>
`/hpeq/passthrough true`, `/hpeq/passthrough false`, `/hpeq/passthrough toggle`
* The target name is derived from the individual _JACK_ client name for all commands, while the order of target client and command can be altered. Additional commands might be available.<br/>
`/renderer/crossfade`, `/crossfade/renderer`<br/>
`/tracker/zero`, `/tracker/azimuth 45`, `/player/stop`, `/player/play`, `/quit`
* In the package included is an example remote control client implemented for [_"vanilla" PD_](http://puredata.info/), see further instructions in [OSC_Remote_Demo.pd](res/OSC_Remote_Demo.pd).

## Validation - Setup and Execution:
* Download and build required [_ecasound_ library](https://ecasound.seul.org/ecasound/download.php) for signal playback and capture __with _JACK_ support__<br/>
in directory `./configure`, `make` and `sudo make install` while having _JACK_ installed
* __Optional:__ Install [_sendosc_](https://github.com/yoggy/sendosc) tool to be used for automation in shell scripts<br/>
`brew install yoggy/tap/sendosc`
* __Remark:__ Make sure all subsequent rendering configurations are able to start up properly before recording starts (particularly FFTW optimization might take a long time, see above)
* Validate impulse responses by **comparing against a reference implementation**, in this case the output of [_sound_field_analysis-py_](https://nbviewer.jupyter.org/github/AppliedAcousticsChalmers/sound_field_analysis-py/blob/master/examples/Exp4_BinauralRendering.ipynb) [[7]](#references)
  * Execute recording script, consecutively starting the package and capturing impulse responses in different rendering configurations<br/>
  `./res/validation/record_ir.sh`<br/>
  __Remark:__ Both implementations compensate the source being at an incidence angle of -37 degrees in the measurement IR set
  * Run package in validation mode, executing a comparison of all beforehand captured IRs in `res/validation/` against the provided reference IRs<br/>
  `python -m ReTiSAR --VALIDATION_MODE=res/BRIR/CR1_VSA_110RS_L_SSR_SFA_-37_SOFA_RFI.wav`
* Validate signal-to-noise-ratio by **comparing input and output signals of the main binaural renderer for wanted target signals and emulated sensor self-noise** respectively 
  * Execute recording script consecutively starting the package and capturing target-noise as well as self-noise input and output signals in different rendering configurations<br/>
  `./res/validation/record_snr.sh`
  * Open (and run) _MATLAB_ analysis script to execute an SNR comparison of beforehand captured signals<br/>
  `open ./res/validation/calculate_snr.m`

## Benchmark - Setup and Execution:
* Install addition required _Python_ packages into _Conda_ environment<br/>
`conda env update --file environment_dev.yml `
* Run the _JACK_ server with sampling rate of __44100 Hz__ via _JackPilot_ or open a new command line window `[CMD]+[T]` and<br/>
`jackd -d coreaudio -r 44100`
* Run package in benchmark mode, instantiating one rendering _JACK_ client with as many convolver instances as possible (35-60 minutes)<br/>
`python -m ReTiSAR --BENCHMARK_MODE=PARALLEL_CONVOLVERS`
* Run package in benchmark mode, instantiating as many rendering _JACK_ clients as possible with one convolver instance (10-15 minutes)<br/>
`python -m ReTiSAR --BENCHMARK_MODE=PARALLEL_CLIENTS`
* Find generated results in the specified files at the end of the script.

## References
[[1]](https://research.chalmers.se/en/publication/509494) Helmholz, H., Andersson, C., and Ahrens, J. (2019). “Real-Time Implementation of Binaural Rendering of High-Order Spherical Microphone Array Signals,” Fortschritte der Akust. -- DAGA 2019, Deutsche Gesellschaft für Akustik, Rostock, Germany, 1462-1465.<br/>
[[2]](http://audiogroup.web.th-koeln.de/PUBLIKATIONEN/Bernschuetz_DAGA2011_01.pdf) Bernschütz, B., Pöschmann, C., Spors, S., and Weinzierl, S. (2011). “Soft-Limiting der modalen Amplitudenverstärkung bei sphärischen Mikrofonarrays im Plane Wave Decomposition Verfahren,” Fortschritte der Akust. -- DAGA 2011, Deutsche Gesellschaft für Akustik, Düsseldorf, Germany, 661–662.<br/>
[[3]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8683751&tag=1) Hold, C., Gamper, H., Pulkki, V., Raghuvanshi, N., and Tashev, I. J. (2019). “Improving Binaural Ambisonics Decoding by Spherical Harmonics Domain Tapering and Coloration Compensation,” Int. Conf. Acoust. Speech Signal Process., IEEE, Brighton, UK, 261–265. doi:10.1109/ICASSP.2019.8683751<br/>
[[4]](http://asa.scitation.org/doi/10.1121/1.4983652) Ben-Hur, Z., Brinkmann, F., Sheaffer, J., Weinzierl, S., and Rafaely, B. (2017). “Spectral equalization in binaural signals represented by order-truncated spherical harmonics,” J. Acoust. Soc. Am., 141, 4087–4096. doi:10.1121/1.4983652<br/>
[[5]](http://www.audiogroup.web.fh-koeln.de/FILES/AIA-DAGA2013_HRIRs.pdf) Bernschütz, B. (2013). “A spherical far field HRIR/HRTF compilation of the Neumann KU 100,” Fortschritte der Akust. -- AIA/DAGA 2013, Deutsche Gesellschaft für Akustik, Meran, Italy, 592–595.<br/>
[[6]](http://www.aes.org/e-lib/browse.cfm?elib=16781) Majdak, P., Iwaya, Y., Carpentier, T., Nicol, R., Parmentier, M., Roginska, A., Suzuki, Y., et al. (2013). “Spatially Oriented Format for Acoustics: A Data Exchange Format Representing Head-Related Transfer Functions,” AES Conv. 134, Audio Engineering Society, Rome, 262–272.<br/>
[[7]](http://www.audiogroup.web.fh-koeln.de/FILES/VDT2012_WDRIRC.pdf) Stade, P., Bernschütz, B., and Rühl, M. (2012). “A Spatial Audio Impulse Response Compilation Captured at the WDR Broadcast Studios,” 27th Tonmeistertagung -- VDT Int. Conv., Verband Deutscher Tonmeister e.V., Cologne, Germany, 551–567.<br/>
[[8]](https://pdfs.semanticscholar.org/3c9a/ed0153b9eb94947953ddb326c3de29ae5f75.pdf) Hohnerlein, C., and Ahrens, J. (2017). “Spherical Microphone Array Processing in Python with the sound field analysis-py Toolbox,” Fortschritte der Akust. -- DAGA 2017, Deutsche Gesellschaft für Akustik, Kiel, Germany, 1033–1036.<br/>

## Changelog
...

## License
This software is licensed under a Non-Commercial Software License (see [LICENSE](LICENSE) for full details).

Copyright (c) 2018<br/>
Division of Applied Acoustics<br/>
Chalmers University of Technology
