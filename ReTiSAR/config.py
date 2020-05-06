from . import tools


# ################## #
#  DEFAULT SETTINGS  #
# ################## #

BLOCK_LENGTH = 4096
# BLOCK_LENGTH = 2048
# BLOCK_LENGTH = 1024
# BLOCK_LENGTH = 512
# BLOCK_LENGTH = 256
"""[optional] Block length of the JACK audio server and clients in samples, see `JackClient`. Should only be set before
starting any client."""

REMOTE_OSC_PORT = 5005
"""[optional] Port to receive Open Sound Control remote control messages."""

TRACKER_TYPE = 'AUTO_ROTATE'
"""[optional] Type of hardware providing head tracking data, see ``HeadTracker.Type`."""
# TRACKER_TYPE = 'POLHEMUS_PATRIOT'
# """[optional] Type of hardware providing head tracking data, see `HeadTracker.Type`."""
# TRACKER_PORT = '/dev/tty.UC-232AC'
# """[optional] System specific path to tracker port to read data from, see `HeadTracker`."""
# TRACKER_TYPE = 'RAZOR_AHRS'
# """[optional] Type of hardware providing head tracking data, see `HeadTracker.Type`."""
# TRACKER_PORT = '/dev/tty.usbserial-AH03F9XC'
# """[optional] System specific path to tracker port to read data from, see `HeadTracker`."""

G_TYPE = 'NOISE_IIR_PINK'
# G_TYPE = 'NOISE_AR_PINK'
# G_TYPE = 'NOISE_WHITE'
# G_TYPE = 'IMPULSE_DIRAC'
"""[optional] Type of algorithm used by generator to create the specified sound, see `Generator.Type`."""
G_LEVEL = -20
"""[optional] Output level of sound generator, see `JackGenerator`."""
G_MUTE = True
"""[optional] Output mute state of sound generator, see `JackGenerator`."""

SOURCE_FILE = 'res/source/Drums_48.wav'
# SOURCE_FILE = 'res/source/Sine500_48.wav'
"""[optional] File of audio being played by the application, see `JackPlayer`."""
# SOURCE_LEVEL = -3
# """[optional] Output level of audio being played by the application, see `JackClient`."""
# SOURCE_MUTE = True
# """[optional] Output mute state of audio being played by the application, see `JackClient`."""

ARIR_TYPE = 'ARIR_MIRO'
"""[optional] Output level of renderer for Array Room Impulse Responses, see `JackClient`."""
# ARIR_LEVEL = -12
# """[optional] Output level of renderer for Array Room Impulse Responses, see `JackClient`."""
# ARIR_MUTE = True
# """[optional] Output mute state of renderer for Array Room Impulse Responses, see `JackClient`."""
ARIR_RADIAL_AMP = 18
# ARIR_RADIAL_AMP = 0
"""[optional] Maximum amplification limit in dB when generating modal radial filters, see `FilterSet`."""

HRIR_FILE = 'res/HRIR/HRIR_L2702_struct.mat'
"""File with FIR filter containing Head Related Impulse Responses, see `FilterSet`."""
HRIR_TYPE = 'HRIR_MIRO'
"""Type of FIR filter file containing Head Related Impulse Responses, see `FilterSet.Type`."""
# HRIR_LEVEL = -3
# """[optional] Output level of renderer for Head Related Impulse Responses, see `JackClient`."""
# HRIR_MUTE = True
# """[optional] Output mute state of renderer for Head Related Impulse Responses, see `JackClient`."""
# HRIR_SUBSONIC = 20  # EXPERIMENTAL!
# """[optional] Subsonic filter cutoff frequency renderer for Head Related Impulse Responses, see `Convolver`."""

# HPIR_FILE = 'res/HpIR/FABIAN_TU_Sennheiser_HD650/HpFilter.wav'
# """[optional] File with FIR filter containing Headphone Equalization Impulse Responses, see `FilterSet`."""
HPIR_TYPE = 'FIR_MULTICHANNEL'
"""[optional] Type of FIR filter file containing Headphone Equalization Impulse Responses, see `FilterSet.Type`."""
# HPIR_LEVEL = -3
# """[optional] Output level of renderer for Headphone Equalization Impulse Responses, see `JackClient`."""
# HPIR_MUTE = True
# """[optional] Output mute state of renderer for Headphone Equalization Impulse Responses, see `JackClient`."""


# ######################### #
#  OPERATION MODE SETTINGS  #
# ######################### #

# ############ ARRAY IR SET RENDERING OF COLOGNE 110 CHANNEL ############
# SOURCE_FILE = 'res/source/Drums_48.wav'
# """[optional] File of audio being played by the application, see `JackPlayer`."""
# SOURCE_POSITIONS = [(-37, 0)]
# """[optional, in case related to ARIR] reference frontal position as list of tuple of azimuth (counterclockwise) and
# elevation in degrees (int or float)."""
ARIR_FILE = 'res/ARIR/CR1_VSA_110RS_L_struct.mat'
"""[optional] File with FIR filter containing Array Room Impulse Responses, see `FilterSet`."""
# ARIR_TYPE = 'ARIR_MIRO'
# """[optional] Type of FIR filter file containing Array Room Impulse Responses, see `FilterSet.Type`."""
ARIR_LEVEL = -12
"""[optional] Output level of renderer for Array Room Impulse Responses, see `JackClient`."""
SH_MAX_ORDER = 8
"""[optional] Maximum spherical harmonics order when rendering Array Room Impulse Responses, see `JackRenderer`."""
# noinspection PyRedeclaration
HRIR_FILE = 'res/HRIR/HRIR_L2702_eq_CR1_VSA_110RS_L_struct.mat'
"""File with FIR filter containing Head Related Impulse Responses, see `FilterSet`."""
# HRIR_TYPE = 'HRIR_MIRO'
# """Type of FIR filter file containing Head Related Impulse Responses, see `FilterSet.Type`."""
HRIR_LEVEL = 6
"""[optional] Output level of renderer for Head Related Impulse Responses, see `JackClient`."""


# ############ ARRAY IR SET RENDERING OF COLOGNE 50 CHANNEL ############
# # SOURCE_FILE = 'res/source/Drums_48.wav'
# # """[optional] File of audio being played by the application, see `JackPlayer`."""
# SOURCE_POSITIONS = [(-37, 0)]
# """[optional, in case related to ARIR] reference frontal position as list of tuple of azimuth (counterclockwise) and
# elevation in degrees (int or float)."""
# ARIR_FILE = 'res/ARIR/CR1_VSA_50RS_L_struct.mat'
# """[optional] File with FIR filter containing Array Room Impulse Responses, see `FilterSet`."""
# # ARIR_TYPE = 'ARIR_MIRO'
# # """[optional] Type of FIR filter file containing Array Room Impulse Responses, see `FilterSet.Type`."""
# ARIR_LEVEL = -12
# """[optional] Output level of renderer for Array Room Impulse Responses, see `JackClient`."""
# SH_MAX_ORDER = 5
# """[optional] Maximum spherical harmonics order when rendering Array Room Impulse Responses, see `JackRenderer`."""
# HRIR_FILE = 'res/HRIR/HRIR_L2702_eq_CR1_VSA_50RS_L_struct.mat'
# """File with FIR filter containing Head Related Impulse Responses, see `FilterSet`."""
# # HRIR_TYPE = 'HRIR_MIRO'
# # """Type of FIR filter file containing Head Related Impulse Responses, see `FilterSet.Type`."""
# HRIR_LEVEL = 6
# """[optional] Output level of renderer for Head Related Impulse Responses, see `JackClient`."""


# ############ ARRAY IR SET RENDERING OF EIGENMIKE ############
# # SOURCE_FILE = 'res/source/Drums_48.wav'
# # """[optional] File of audio being played by the application, see `JackPlayer`."""
# ARIR_FILE = 'res/ARIR/Eigenmike_synthetic_struct.mat'
# # ARIR_FILE = 'res/ARIR/Eigenmike_CH_anechoic_struct.mat'
# """[optional] File with FIR filter containing Array Room Impulse Responses, see `FilterSet`."""
# # ARIR_TYPE = 'ARIR_MIRO'
# # """[optional] Output level of renderer for Array Room Impulse Responses, see `JackClient`."""
# SH_MAX_ORDER = 4
# """[optional] Maximum spherical harmonics order when rendering Array Room Impulse Responses, see `JackRenderer`."""
# HRIR_FILE = 'res/HRIR/HRIR_L2702_eq_Eigenmike_struct.mat'
# """File with FIR filter containing Head Related Impulse Responses, see `FilterSet`."""
# # HRIR_TYPE = 'HRIR_MIRO'
# # """Type of FIR filter file containing Head Related Impulse Responses, see `FilterSet.Type`."""
# HRIR_LEVEL = -30
# """[optional] Output level of renderer for Head Related Impulse Responses, see `JackClient`."""


# ############ ARRAY RECORDING RENDERING OF EIGENMIKE ############
# SOURCE_FILE = 'res/source/Eigenmike_CH_LL_around.wav'
# # SOURCE_FILE = 'res/source/Eigenmike_CH_LL_updown.wav'
# """[optional] File of audio being played by the application, see `JackPlayer`."""
# TRACKER_TYPE = 'NONE'
# """[optional] Type of hardware providing head tracking data, see ``HeadTracker.Type`."""
# ARIR_FILE = 'res/ARIR/Eigenmike_CH_calibration_struct.mat'
# """[optional] File with FIR filter containing Array Room Impulse Responses, see `FilterSet`."""
# ARIR_TYPE = 'AS_MIRO'
# """[optional] Type of FIR filter file containing Array Room Impulse Responses, see `FilterSet.Type`."""
# SH_MAX_ORDER = 4
# """[optional] Maximum spherical harmonics order when rendering Array Room Impulse Responses, see `JackRenderer`."""
# HRIR_FILE = 'res/HRIR/HRIR_L2702_eq_Eigenmike_struct.mat'
# """File with FIR filter containing Head Related Impulse Responses, see `FilterSet`."""
# # HRIR_TYPE = 'HRIR_MIRO'
# # """Type of FIR filter file containing Head Related Impulse Responses, see `FilterSet.Type`."""
# HRIR_LEVEL = -30
# """[optional] Output level of renderer for Head Related Impulse Responses, see `JackClient`."""


# ############ ARRAY REAL-TIME RENDERING OF EIGENMIKE ############
# TRACKER_TYPE = 'NONE'
# """[optional] Type of hardware providing head tracking data, see ``HeadTracker.Type`."""
# SOURCE_FILE = 'NONE'
# """[optional] File of audio being played by the application, see `JackPlayer`."""
# ARIR_FILE = 'res/ARIR/Eigenmike_CH_calibration_struct.mat'
# # ARIR_FILE = 'res/ARIR/Eigenmike_OC_calibration_struct.mat'
# """[optional] File with FIR filter containing Array Room Impulse Responses, see `FilterSet`."""
# ARIR_TYPE = 'AS_MIRO'
# """[optional] Type of FIR filter file containing Array Room Impulse Responses, see `FilterSet.Type`."""
# SH_MAX_ORDER = 4
# """[optional] Maximum spherical harmonics order when rendering Array Room Impulse Responses, see `JackRenderer`."""
# HRIR_FILE = 'res/HRIR/HRIR_L2702_eq_Eigenmike_struct.mat'
# """File with FIR filter containing Head Related Impulse Responses, see `FilterSet`."""
# # HRIR_TYPE = 'HRIR_MIRO'
# # """Type of FIR filter file containing Head Related Impulse Responses, see `FilterSet.Type`."""
# HRIR_LEVEL = -30
# """[optional] Output level of renderer for Head Related Impulse Responses, see `JackClient`."""


# ############ BRIR SET RENDERING OF SOUND_FIELD_ANALYSIS OUTPUT ############
# # SOURCE_FILE = 'res/source/Drums_48.wav'
# # """[optional] File of audio being played by the application, see `JackPlayer`."""
# HRIR_FILE = 'res/BRIR/CR1_VSA_110RS_L_SSR_SFA_-37.wav'
# """File with FIR filter containing Head Related Impulse Responses, see `FilterSet`."""
# HRIR_TYPE = 'BRIR_SSR'
# """Type of FIR filter file containing Head Related Impulse Responses, see `FilterSet.Type`."""
# HRIR_LEVEL = -12
# """[optional] Output level of renderer for Head Related Impulse Responses, see `JackClient`."""


# ############ HRIR SET RENDERING OF ARBITRARY SOURCE NUMBER at 44100 Hz ############
# SOURCE_FILE = 'res/source/PinkMartini_Lilly.wav'
# """[optional] File of audio being played by the application, see `JackPlayer`."""
# SOURCE_POSITIONS = [(30, 0), (-30, 0)]
# """Source positions as list of tuple of azimuth (counterclockwise) and elevation in degrees (int or float)."""
# HRIR_FILE = 'res/HRIR/FABIAN_TU/hrirs_fabian.wav'
# # HRIR_FILE = 'res/HRIR/SSR_ID-46.wav'
# """File with FIR filter containing Head Related Impulse Responses, see `FilterSet`."""
# HRIR_TYPE = 'HRIR_SSR'
# """Type of FIR filter file containing Head Related Impulse Responses, see `FilterSet.Type`."""
# # HRIR_FILE = 'res/HRIR/KEMAR_MIT/mit_kemar_large_pinna.sofa'
# # """File with FIR filter containing Head Related Impulse Responses, see `FilterSet`."""
# # HRIR_TYPE = 'HRIR_SOFA'
# # """Type of FIR filter file containing Head Related Impulse Responses, see `FilterSet.Type`."""


# ###################### #
#  PERFORMANCE SETTINGS  #
# ###################### #

# PROCESS_PRIORITY = 1
# """[optional] Process attribute, which is requested as its operation system specific priority, see `__main__`."""

PROCESS_FILE_LIMIT_MIN = 10240
"""[optional] MacOS specific process attribute to limit the number of open files, which is requested to be as high as 
the specified value."""

NUMPY_OMP_NUM_THREADS = 1  # 1 leads to best performance so far
"""Number of available OpenMP threads, should be chosen carefully to not lead to oversubscription (1 is equivalent to 
disabling threading in OpenMP)."""
NUMPY_OMP_DYNAMIC = False  # False leads to best performance so far
"""Enable dynamic reduction of number of threads based on analysis of system workload, which may reduce possible 
oversubscription from OpenMP threading."""
NUMPY_MKL_NUM_THREADS = int(tools.get_cpu_count() / 4)  # quarter CPU count leads to best real-time performance so far
"""Number of available MKL threads, should be chosen carefully to not lead to oversubscription (1 is equivalent to 
disabling threading in MKL)."""
NUMPY_MKL_DYNAMIC = False  # False leads to best performance so far
"""Enable dynamic reduction of number of threads based on analysis of system workload, which may reduce possible 
oversubscription from MKL threading."""

IS_PYFFTW_MODE = True  # True leads to best performance so far
"""If `pyfftw` package (wrapper for FFTW library) should be used instead of `numpy` for all real-time DFT operations. In
case `pyfftw` is not used, all related tasks like loading/saving and pre-calculating FFTW wisdom will be skipped."""
# PYFFTW_EFFORT = 'FFTW_ESTIMATE'
# PYFFTW_EFFORT = 'FFTW_MEASURE'
PYFFTW_EFFORT = 'FFTW_PATIENT'
# PYFFTW_EFFORT = 'FFTW_EXHAUSTIVE'
"""[optional] Amount of effort spent during the FFTW planning stage to create the fastest possible transform, see
`pyfftw`."""
PYFFTW_NUM_THREADS = int(tools.get_cpu_count() / 4)  # quarter CPU count leads to best real-time performance so far
"""[optional] Number of available FFTW threads, should be chosen carefully to not lead to oversubscription (1 is
equivalent to disabling threading in FFTW)."""
PYFFTW_WISDOM_FILE = 'pyfftw_wisdom.bin'
"""File of `pyfftw` wisdom being loaded/saved by the application, see `__main__`."""


# ################## #
#  LOGGING SETTINGS  #
# ################## #

# LOGGING_LEVEL = 'DEBUG'
LOGGING_LEVEL = 'INFO'
"""Lowest level of messages being printed to the logs, see `process_logger`."""

# LOGGING_FORMAT = '%(processName)s  %(levelname)-8s  %(message)s'
# LOGGING_FORMAT = '%(name)s  %(levelname)-8s  %(message)s'
LOGGING_FORMAT = '%(name)-@s  %(levelname)-8s  %(message)s'
"""Format of messages being printed to the log, see `process_logger`."""

LOGGING_PATH = 'log/'
"""[optional] Path of log messages being saved to, see `process_logger`."""


# ################# #
#  ! DO NOT EDIT !  #
# ################# #

IS_DEBUG_MODE = tools.get_is_debug()
"""If the application is run in a debugging mode. When execution is paused this is used so certain processes relying 
on real time execution do not raise errors. Also this is used to make use of breakpoints in certain functions before 
they get released in a separate process."""

BENCHMARK_MODE = False
"""Run mode specifying a benchmarking method, see `tools` command line parsing."""
VALIDATION_MODE = False
"""Run mode specifying a validation method against provided reference impulse response set, see `tools` command line
parsing."""
DEVELOPER_MODE = False
"""Run mode specifying a development test method, see `tools` command line parsing."""

if 'BLOCK_LENGTH' not in locals():
    BLOCK_LENGTH = None
if 'SH_MAX_ORDER' not in locals():
    SH_MAX_ORDER = None
if 'ARIR_RADIAL_AMP' not in locals():
    ARIR_RADIAL_AMP = 0
if 'HRIR_SUBSONIC' not in locals():
    HRIR_SUBSONIC = None
if 'SOURCE_POSITIONS' not in locals():
    SOURCE_POSITIONS = [(0, 0)]
if 'TRACKER_TYPE' not in locals():
    TRACKER_TYPE = None
if 'TRACKER_PORT' not in locals():
    TRACKER_PORT = None
if 'REMOTE_OSC_PORT' not in locals():
    REMOTE_OSC_PORT = None
if 'G_TYPE' not in locals():
    G_TYPE = None
if 'ARIR_TYPE' not in locals():
    ARIR_TYPE = None
if 'HPIR_TYPE' not in locals():
    HPIR_TYPE = None
if 'SOURCE_LEVEL' not in locals():
    SOURCE_LEVEL = 0
if 'G_LEVEL' not in locals():
    G_LEVEL = 0
if 'ARIR_LEVEL' not in locals():
    ARIR_LEVEL = 0
if 'HRIR_LEVEL' not in locals():
    HRIR_LEVEL = 0
if 'HPIR_LEVEL' not in locals():
    HPIR_LEVEL = 0
if 'SOURCE_MUTE' not in locals():
    SOURCE_MUTE = False
if 'G_MUTE' not in locals():
    G_MUTE = False
if 'ARIR_MUTE' not in locals():
    ARIR_MUTE = False
if 'HRIR_MUTE' not in locals():
    HRIR_MUTE = False
if 'HPIR_MUTE' not in locals():
    HPIR_MUTE = False

# transform following attributes from relative package into absolute system paths
HRIR_FILE = tools.get_absolute_from_relative_package_path(HRIR_FILE)
# noinspection PyUnboundLocalVariable,PyUnresolvedReferences
ARIR_FILE = tools.get_absolute_from_relative_package_path(ARIR_FILE) if 'ARIR_FILE' in locals() else None
# noinspection PyUnboundLocalVariable,PyUnresolvedReferences
SOURCE_FILE = tools.get_absolute_from_relative_package_path(SOURCE_FILE) if 'SOURCE_FILE' in locals() else None
# noinspection PyUnboundLocalVariable,PyUnresolvedReferences
HPIR_FILE = tools.get_absolute_from_relative_package_path(HPIR_FILE) if 'HPIR_FILE' in locals() else None
# noinspection PyUnboundLocalVariable,PyUnresolvedReferences
LOGGING_PATH = tools.get_absolute_from_relative_package_path(LOGGING_PATH) if 'LOGGING_PATH' in locals() else None
if LOGGING_PATH:
    PYFFTW_WISDOM_FILE = LOGGING_PATH + PYFFTW_WISDOM_FILE
