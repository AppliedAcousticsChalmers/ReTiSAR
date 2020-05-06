# ReTiSAR
Implementation of the Real-Time Spherical Microphone Renderer for binaural reproduction in _Python_.

## Requirements:
* _MacOS_ (other OS not tested)
* _JACK_ library, see http://jackaudio.org/downloads/ (on _MacOS_ the prebuilt binary works best)
* _Conda_ installation, see https://conda.io/en/master/miniconda.html (`miniconda` is enough, highly recommended to get Intel-optimized _MKL_ `numpy` version)
* _Python_ installation, recommended way is to use _Conda_ (automatically done in setup section below)
* Installation of the required _Python_ packages (see setup section below)
* __Optional:__ _OSC_ client, see http://opensoundcontrol.org/implementations (see remote control section below)

## Checkout / Download:
* Clone _git_ repository via command line<br/>
`git clone https://github.com/AppliedAcousticsChalmers/ReTiSAR.git`
* __Alternative 1:__ Clone via any other _git_ client<br/>
_GitHub desktop client_ worked well after login, since login credentials on _MacOS_ command line did not work … presumably because of 2FA-authentication
* __Alternative 2:__ Download and extract manually from provided URL

## Setup:
* Navigate into repository<br/>
`cd ReTiSAR/`
* Make sure that _Conda_ is up to date<br/>
`conda update conda`
* __Optional:__ Delete existing _Conda_ environment<br/>
`conda env remove -n ReTiSAR`
* Create new _Conda_ environment from the specified requirements<br/>
`conda env create -f environment.yml`
* Activate _Conda_ environment<br/>
`source activate ReTiSAR`

## Execution:
* __Optional:__ In case you have never started the _JACK_ audio server on your system or want to make sure it initializes with appropriate values. Open the _JackPilot_ application set your system specific default settings.<br/>
At this point the only relevant _JACK_ audio server setting is the sampling frequency, which has to match the sampling frequency of your rendered audio source file or stream (no resampling will be applied for that specific file).<br/>
All here provided examples are supposed to be run at a sampling frequency of __48 kHz__.
* Run package with default parameters<br/>
`python -m ReTiSAR`
* Show possible configuration parameters to run the application<br/>
`python -m ReTiSAR --help`

The following parameters are all optional and available in combinations with the named configurations subsequently:<br/>
* Run package with possibly lower block size for decreased latency (choose value according to individual rendering modes and performance of your system)<br/>
`python -m ReTiSAR -b=512`
* Run package with individual head-tracking device (paths system dependent)<br/>
`python -m ReTiSAR -tt=AUTO_ROTATE` (which is the default behaviour) or<br/>
`python -m ReTiSAR -tt=RAZOR_AHRS -t=/dev/tty.usbserial-AH03F9XC` or<br/>
`python -m ReTiSAR -tt=POLHEMUS_PATRIOT -t=/dev/tty.UC-232AC`
* Run package with individual amplification limit in dB when generating modal radial filters (in case of spherical harmonics processing)<br/>
`python -m ReTiSAR -arr=18`
* Run package with individual headphone equalization (partitioned convolution)<br/>
`python -m ReTiSAR -hp=res/HpIR/FABIAN_TU_Sennheiser_HD650/HpFilter.wav`

Specific configurations:<br/>
* Run package as array stream renderer of _Eigenmike_ (**Oc**ulus or **Ch**almers)<br/>
`python -m ReTiSAR -sh=4 -tt=NONE -s=None -ar=res/ARIR/Eigenmike_OC_calibration_struct.mat -art=AS_MIRO  -hr=res/HRIR/HRIR_L2702_eq_Eigenmike_struct.mat -hrt=HRIR_MIRO -hrl=-30` or<br/>
`python -m ReTiSAR -sh=4 -tt=NONE -s=None -ar=res/ARIR/Eigenmike_CH_calibration_struct.mat -art=AS_MIRO  -hr=res/HRIR/HRIR_L2702_eq_Eigenmike_struct.mat -hrt=HRIR_MIRO -hrl=-30`
* Run package as array recording renderer of _Eigenmike_<br/>
`python -m ReTiSAR -sh=4 -tt=NONE -s=res/source/Eigenmike_CH_LL_around.wav -ar=res/ARIR/Eigenmike_CH_calibration_struct.mat -art=AS_MIRO -hr=res/HRIR/HRIR_L2702_eq_Eigenmike_struct.mat -hrt=HRIR_MIRO -hrl=-30` or<br/>
`python -m ReTiSAR -sh=4 -tt=NONE -s=res/source/Eigenmike_CH_LL_updown.wav -ar=res/ARIR/Eigenmike_CH_calibration_struct.mat -art=AS_MIRO -hr=res/HRIR/HRIR_L2702_eq_Eigenmike_struct.mat -hrt=HRIR_MIRO -hrl=-30`

* Run package as array IR renderer of _Eigenmike_<br/>
`python -m ReTiSAR -sh=4 -s=res/source/Drums_48.wav -ar=res/ARIR/Eigenmike_synthetic_struct.mat -art=ARIR_MIRO -hr=res/HRIR/HRIR_L2702_eq_Eigenmike_struct.mat -hrt=HRIR_MIRO -hrl=-30` or<br/>
`python -m ReTiSAR -sh=4 -s=res/source/Drums_48.wav -ar=res/ARIR/Eigenmike_CH_anechoic_struct.mat -art=ARIR_MIRO -hr=res/HRIR/HRIR_L2702_eq_Eigenmike_struct.mat -hrt=HRIR_MIRO -hrl=-30`
* Run package as array IR renderer with 50 channels corresponding to 5th order<br/>
`python -m ReTiSAR -sh=5 -s=res/source/Drums_48.wav -sp="[(-37,0)]" -ar=res/ARIR/CR1_VSA_50RS_L_struct.mat -art=ARIR_MIRO -arl=-12 -hr=res/HRIR/HRIR_L2702_eq_CR1_VSA_50RS_L_struct.mat -hrt=HRIR_MIRO -hrl=6`
* Run package as array IR renderer with 110 channels corresponding to 8th order<br/>
`python -m ReTiSAR -sh=8 -s=res/source/Drums_48.wav -sp="[(-37,0)]" -ar=res/ARIR/CR1_VSA_110RS_L_struct.mat -art=ARIR_MIRO -arl=-12 -hr=res/HRIR/HRIR_L2702_eq_CR1_VSA_110RS_L_struct.mat -hrt=HRIR_MIRO -hrl=6`

* Run package as BRIR renderer with partitioned convolution in frequency domain<br/>
`python -m ReTiSAR -s=res/source/Drums_48.wav -hr=res/BRIR/CR1_VSA_110RS_L_SSR_SFA_-37.wav -hrt=BRIR_SSR -hrl=-12`
* Run package as "binauralizer" (rendering of virtual sound sources over HRTF) with partitioned convolution in frequency domain<br/>
sampling frequency at __44100 Hz__ in this case<br/>
`python -m ReTiSAR -s=res/source/PinkMartini_Lilly.wav -sp="[(30, 0),(-30, 0)]" -hr=res/HRIR/FABIAN_TU/hrirs_fabian.wav -hrt=HRIR_SSR`

## Remote Control:
* Certain parameters of the running real-time application can be remote controlled via _Open Sound Control_. Individual clients can be accessed by targeting them with specific _OSC_ commands.<br/>
Depending on the current configuration and rendering mode different commands will be available, i.e. arbitrary combinations of the following.<br/>
`/generator/volume 0`, `/generator/volume -12`<br/>
`/prerenderer/mute 1`, `/prerenderer/mute 0`, `/prerenderer/mute -1`, `/prerenderer/mute`<br/>
`/hpeq/passthrough true`, `/hpeq/passthrough false`, `/hpeq/passthrough toggle`
* The target name is derived from the individual _JACK_ client name for all commands, while the order of target client and command can be altered. Additional commands might be available.<br/>
`/renderer/crossfade`, `/crossfade/renderer`<br/>
`/tracker/zero`, `/tracker/azimuth 45`, `/player/stop`, `/player/play`, `/quit`
* In the package included is an example remote control client implemented for _"vanilla" PD_ (http://puredata.info/), see further instructions in `./res/OSC_Remote_Demo.pd`.

## Validation IR - Setup and Execution:
* Download and build required _ecasound_ library for signal playback and capture __with _JACK_ support__, see https://ecasound.seul.org/ecasound/download.php<br/>
in directory `./configure`, `make` and `sudo make install` while having _JACK_ installed
* Execute rendering pipeline in identical configuration like compared implementation, in this case the output of _sound_field_analysis-py_ (except for levels)<br/>
i.e. https://nbviewer.jupyter.org/github/AppliedAcousticsChalmers/sound_field_analysis-py/blob/master/examples/Exp4_BinauralRendering.ipynb<br/>
`python -m ReTiSAR -b=4096 -sh=8 -arr=0 -tt=NONE -sp="[(-37,0)]" -ar=res/ARIR/CR1_VSA_110RS_L_struct.mat -art=ARIR_MIRO -arl=-12 -hr=res/HRIR/HRIR_L2702_struct.mat -hrt=HRIR_MIRO -hrl=6 -sm=TRUE -gm=TRUE`<br/>
__Remark:__ both implementations compensate the source being at an incidence angle of -37 degrees in the measurement IR set
* __Optional:__ Set individual head rotation, i.e. 80 degrees via remote client (see remote control section)
* Start _ecasound_ signal capture, impulse playback and signal capture interrupt in a separate terminal window (one combined command)<br/>
`ecasound -f:s32 -i jack,ReTiSAR-Renderer -o res/validation/Impulse_48_result_4096_80.wav & sleep 1 && ecasound -i res/source/Impulse_48.wav -o jack,ReTiSAR-PreRenderer && sleep 1 && kill %1 && sleep 1`<br/>
__Remark:__ change recorded file name according to chosen head rotation (name will be used by validation script) __!__
* Execute validation script by providing a reference, automatically comparing it against all IRs in `res/validation/`<br/>
`python -m ReTiSAR --VALIDATION_MODE=res/BRIR/CR1_VSA_110RS_L_SSR_SFA_-37.wav`

## Validation SNR - Setup and Execution:
* Download and build required _ecasound_ library for signal playback and capture __with _JACK_ support__, see https://ecasound.seul.org/ecasound/download.php<br/>
in directory `./configure`, `make` and `sudo make install` while having _JACK_ installed
* __Optional:__ Install _sendosc_ tool to be used for automation in shell scripts, see https://github.com/yoggy/sendosc<br/>
`brew install yoggy/tap/sendosc`
* Execute rendering pipeline with desired configurations and execute recording scripts with desired parameters (recorded microphone channel; file set name)<br/>
__Remark:__ the recording script contains automated separate measurements of noise and drums signals before and after the rendering pipeline for 0 and 90 degrees head rotation
* i.e. with default parameters<br/>
`python -m ReTiSAR -b=4096 -sh=8 -arr=0 -tt=NONE -s=NONE -sp="[(-37,0)]" -ar=res/ARIR/CR1_VSA_110RS_L_struct.mat -art=ARIR_MIRO -arl=-12 -hr=res/HRIR/HRIR_L2702_struct.mat -hrt=HRIR_MIRO -hrl=0 -gm=FALSE`<br/>
`res/validation/record.sh 14 rec_4096_sh8_0db`
* i.e. with radial filter limit of 18 dB (instead of 0 dB)<br/>
`python -m ReTiSAR -b=4096 -sh=8 -arr=18 -tt=NONE -s=NONE -sp="[(-37,0)]" -ar=res/ARIR/CR1_VSA_110RS_L_struct.mat -art=ARIR_MIRO -arl=-12 -hr=res/HRIR/HRIR_L2702_struct.mat -hrt=HRIR_MIRO -hrl=0 -gm=FALSE`<br/>
`res/validation/record.sh 14 rec_4096_sh8_18db`
* i.e. with maximum spherical harmonics processing order of 5 (instead of 8)<br/>
`python -m ReTiSAR -b=4096 -sh=5 -arr=0 -tt=NONE -s=NONE -sp="[(-37,0)]" -ar=res/ARIR/CR1_VSA_110RS_L_struct.mat -art=ARIR_MIRO -arl=-12 -hr=res/HRIR/HRIR_L2702_struct.mat -hrt=HRIR_MIRO -hrl=0 -gm=FALSE`<br/>
`res/validation/record.sh 14 rec_4096_sh5_0db`
* i.e. with block length of 1024 samples (instead of 4096 samples)<br/>
`python -m ReTiSAR -b=1024 -sh=8 -arr=0 -tt=NONE -s=NONE -sp="[(-37,0)]" -ar=res/ARIR/CR1_VSA_110RS_L_struct.mat -art=ARIR_MIRO -arl=-12 -hr=res/HRIR/HRIR_L2702_struct.mat -hrt=HRIR_MIRO -hrl=0 -gm=FALSE`<br/>
`res/validation/record.sh 14 rec_1024_sh8_0db`
* i.e. with only one noise generator chanel connected, showing directional dependency<br/>
`python -m ReTiSAR -b=4096 -sh=8 -arr=0 -tt=NONE -s=NONE -sp="[(-37,0)]" -ar=res/ARIR/CR1_VSA_110RS_L_struct.mat -art=ARIR_MIRO -arl=-12 -hr=res/HRIR/HRIR_L2702_struct.mat -hrt=HRIR_MIRO -hrl=0 -gm=FALSE`<br/>
__Remark:__  manually disconnect all other channel connections from noise generator to main renderer (e.g. in _QjackCtl_ or _Patchage_)<br/>
`res/validation/record.sh 14 rec_1mic_4096_sh8_0db`
* run analysis script (Matlab) to generate result plots<br/>
`run_snr.m`

## Benchmark - Setup and Execution:
* Install also required _Python_ packages for _benchmark_<br/>
`pip install .[benchmark]`
* Run the _JACK_ server with sampling rate of __44100 Hz__ via _JackPilot_ or open a new command line window `[CMD]+[T]` and<br/>
`jackd -d coreaudio -r 44100`
* Run package in benchmark mode, instantiating one rendering _JACK_ client with as many convolver instances as possible
(35-50 minutes)<br/>
`python -m ReTiSAR --BENCHMARK_MODE=PARALLEL_CONVOLVERS`
* Run package in benchmark mode, instantiating as many rendering _JACK_ clients as possible with one convolver instance
(10-15 minutes)<br/>
`python -m ReTiSAR --BENCHMARK_MODE=PARALLEL_CLIENTS`
* Find generated results in the specified files at the end of the script.
