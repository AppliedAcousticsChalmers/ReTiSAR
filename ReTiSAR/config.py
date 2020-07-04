from . import mp_context, tools

# fmt: off

# ############################## #
#  DEFAULT EXECUTION PARAMETERS  #
# ############################## #

BLOCK_LENGTH = 4096
# BLOCK_LENGTH = 2048
# BLOCK_LENGTH = 1024
# BLOCK_LENGTH = 512
# BLOCK_LENGTH = 256
"""Block length of the JACK audio server and clients in samples, see `JackClient`. Should only be
set before starting any client. """

# TRACKER_TYPE = None
# TRACKER_TYPE = "AUTO_ROTATE"
"""Type of hardware providing head tracking data, see `HeadTracker.Type`."""
# TRACKER_TYPE = "POLHEMUS_PATRIOT"
# TRACKER_TYPE = "POLHEMUS_FASTRACK"
# """Type of hardware providing head tracking data, see `HeadTracker.Type`."""
# TRACKER_PORT = "/dev/tty.UC-232AC"
# """System specific path to tracker port to read data from, see `HeadTracker`."""
# TRACKER_TYPE = "RAZOR_AHRS"
# """Type of hardware providing head tracking data, see `HeadTracker.Type`."""
# TRACKER_PORT = "/dev/tty.usbserial-AH03F9XC"
"""System specific path to tracker port to read data from, see `HeadTracker`."""

SOURCE_LEVEL = 0
"""Output level in dBFS of audio being played by the application, see `JackClient`."""
SOURCE_MUTE = False
"""Output mute state of audio being played by the application, see `JackClient`."""

G_TYPE = None
# G_TYPE = "NOISE_WHITE"
# G_TYPE = "NOISE_IIR_PINK"
# G_TYPE = "NOISE_IIR_EIGENMIKE"
"""Type of algorithm used by generator to create the specified sound, see `Generator.Type`."""
G_LEVEL = -30
"""Output level in dBFS of sound generator, see `JackGenerator`."""
G_MUTE = False
"""Output mute state of sound generator, see `JackGenerator`."""

# ARIR_RADIAL_AMP = 0
ARIR_RADIAL_AMP = 18
"""Maximum amplification limit in dB when generating modal radial filters, see `FilterSet`."""
ARIR_MUTE = False
"""Output mute state of renderer for Array Room Impulse Responses, see `JackClient`."""

HRIR_TYPE = "HRIR_SOFA"
"""Type of FIR filter file containing Head Related Impulse Responses, see `FilterSet.Type`."""
HRIR_FILE = "res/HRIR/KU100_THK/HRIR_L2702.sofa"
# HRIR_FILE = "res/HRIR/FABIAN_TUB/FABIAN_HRIR_measured_HATO_0.sofa"
"""File with FIR filter containing Head Related Impulse Responses, see `FilterSet`."""
# HRIR_TYPE = "HRIR_MIRO"
# """Type of FIR filter file containing Head Related Impulse Responses, see `FilterSet.Type`."""
# HRIR_FILE = "res/HRIR/KU100_THK/HRIR_L2702_struct.mat"
# """File with FIR filter containing Head Related Impulse Responses, see `FilterSet`."""
HRIR_LEVEL = 0
"""Output level in dBFS of renderer for Head Related Impulse Responses, see `JackClient`."""
HRIR_MUTE = False
"""Output mute state of renderer for Head Related Impulse Responses, see `JackClient`."""
HRIR_DELAY = 0
"""Input buffer delay in ms of renderer for Head Related Impulse Responses, see `JackClient`."""

HPCF_TYPE = "HPCF_FIR"
"""Type of FIR filter file containing Headphone Compensation Filter, see `FilterSet.Type`."""
HPCF_FILE = None
# HPCF_FILE = "res/HPCF/KEMAR_TUR/hpComp_HD600_1Filter.wav"
"""File with FIR filter containing Headphone Compensation Filter, see `FilterSet`."""
HPCF_LEVEL = 0
"""Output level in dBFS of renderer for Headphone Compensation Filter, see `JackClient`."""
HPCF_MUTE = False
"""Output mute state of renderer for Headphone Compensation Filter, see `JackClient`."""

# SH_COMPENSATION_TYPE = "SPHERICAL_HEAD_FILTER"
# SH_COMPENSATION_TYPE = "SPHERICAL_HARMONICS_TAPERING"
SH_COMPENSATION_TYPE = "SPHERICAL_HARMONICS_TAPERING+SPHERICAL_HEAD_FILTER"
"""Type of spherical harmonics processing compensation technique, see `Compensation.Type`."""
SH_IS_ENFORCE_PINV = False
"""If pseudo-inverse (Moore-Penrose) matrix will be used over explicitly given sampling grid
weights (only relevant for filter sets in MIRO format), see `FilterSet`. """

# IR_TRUNCATION_LEVEL = 0  # no truncation
IR_TRUNCATION_LEVEL = -60
# IR_TRUNCATION_LEVEL = -100
"""Level relative under global peak in dB to individually truncate any impulse response set after
load (allows to save performance in case of partitioned convolution), see `FilterSet`. """

CLIENT_MAX_DELAY_SEC = 1
"""Input buffer delay limitation in s for all renderers."""

REMOTE_OSC_PORT = 5005
"""Port to receive Open Sound Control remote control messages."""


# ########################### #
#  EXECUTION MODE PARAMETERS  #
# ########################### #

# [DEFAULT] ############ ARRAY RECORDING RENDERING OF EIGENMIKE ############
# BLOCK_LENGTH = 256
SOURCE_FILE = "res/record/EM32ch_lab_voice_around.wav"  # showcasing horizontal movement
# SOURCE_FILE = "res/record/EM32ch_lab_voice_updown.wav"  # showcasing vertical movement
ARIR_LEVEL = 0
ARIR_TYPE = "AS_MIRO"
SH_MAX_ORDER, ARIR_FILE = 4, "res/ARIR/RT_calib_EM32ch_struct.mat"  # Chalmers (SN 28)

# ############ ARRAY RECORDING RENDERING OF THK HOSMA ############
# BLOCK_LENGTH = 1024
# SOURCE_LEVEL = 9
# SOURCE_POSITIONS, SOURCE_FILE = [(90, 0)], "res/record/HOS64_hall_lecture.wav"  # file not provided
# ARIR_LEVEL = 0
# ARIR_TYPE = "AS_MIRO"
# SH_MAX_ORDER, ARIR_FILE = 7, "res/ARIR/RT_calib_HOS64_struct.mat"

# ############ ARRAY LIVE-STREAM RENDERING ############
# BLOCK_LENGTH = 256
# # BLOCK_LENGTH = 1024
# SOURCE_FILE = None
# ARIR_LEVEL = 0
# ARIR_TYPE = "AS_MIRO"
# SH_MAX_ORDER, ARIR_FILE = 4, "res/ARIR/RT_calib_EM32ch_struct.mat"  # Eigenmike Chalmers (SN 28)
# # SH_MAX_ORDER, ARIR_FILE = 4, "res/ARIR/RT_calib_EM32frl_struct.mat"  # Eigenmike Facebook Reality Labs (SN ??)
# # SH_MAX_ORDER, ARIR_FILE = 7, "res/ARIR/RT_calib_HOS64_struct.mat"  # HØSMA-7N

# ############ ARRAY IR RENDERING OF EIGENMIKE ############
# SOURCE_FILE = "res/source/Drums_48.wav"
# TRACKER_TYPE = "AUTO_ROTATE"  # overwrite in case you want to use head-tracking
# ARIR_LEVEL = -6
# ARIR_TYPE = "ARIR_MIRO"
# SH_MAX_ORDER, ARIR_FILE = 4, "res/ARIR/DRIR_sim_EM32_PW_struct.mat"  # simulated plane wave
# # SH_MAX_ORDER, ARIR_FILE = 4, "res/ARIR/DRIR_anec_EM32ch_S_struct.mat"  # anechoic measurement
# G_LEVEL_REL = [-0.5, -0.1, 1.3, -0.9, -0.3, 1.2, -0.3, -0.8, 0.1, -1.3, -0.5, 0.5, 1.4, -0.7, 0.5,
#                -0.4, 0.2, 1.2, 0.4, -0.3, 0.2, -0.5, 0.0, 0.1, 0.7, 0.8, -0.2, -0.3, -0.2, -0.7,
#                -1.2, 0.6]
# """Output level in dBFS relative between ports of sound generator, see `JackGenerator`.
# Level offsets for noise generator to reproduce relative self-noise level distribution of
# Eigenmike Chalmers (SN 28) according to an anechoic measurement."""

# # ############ ARRAY IR RENDERING OF WDR COLOGNE DATA ############
# SOURCE_FILE = "res/source/Drums_48.wav"
# # SOURCE_POSITIONS = [(-37, 0)]
# # """[in case related to ARIR] reference frontal position as list of tuple of azimuth
# # (counterclockwise) and elevation in degrees (int or float)."""
# TRACKER_TYPE = "AUTO_ROTATE"  # overwrite in case you want to use head-tracking
# ARIR_LEVEL = -12
# ARIR_TYPE = "ARIR_SOFA"
# # SH_MAX_ORDER, ARIR_FILE = 5, "res/ARIR/DRIR_CR1_VSA_50RS_L.sofa"
# # SH_MAX_ORDER, ARIR_FILE = 5, "res/ARIR/DRIR_CR1_VSA_50RS_R.sofa"
# # SH_MAX_ORDER, ARIR_FILE = 5, "res/ARIR/DRIR_CR7_VSA_50RS_L.sofa"
# # SH_MAX_ORDER, ARIR_FILE = 5, "res/ARIR/DRIR_CR7_VSA_50RS_R.sofa"
# # SH_MAX_ORDER, ARIR_FILE = 5, "res/ARIR/DRIR_LBS_VSA_50RS_PAC.sofa"  # example
# # SH_MAX_ORDER, ARIR_FILE = 5, "res/ARIR/DRIR_LBS_VSA_50RS_PAL.sofa"
# # SH_MAX_ORDER, ARIR_FILE = 5, "res/ARIR/DRIR_LBS_VSA_50RS_SBC.sofa"
# # SH_MAX_ORDER, ARIR_FILE = 5, "res/ARIR/DRIR_LBS_VSA_50RS_SBL.sofa"
# # SH_MAX_ORDER, ARIR_FILE = 5, "res/ARIR/DRIR_SBS_VSA_50RS_PAC.sofa"
# # SH_MAX_ORDER, ARIR_FILE = 5, "res/ARIR/DRIR_SBS_VSA_50RS_PAL.sofa"
# # SH_MAX_ORDER, ARIR_FILE = 5, "res/ARIR/DRIR_SBS_VSA_50RS_PAR.sofa"
# # SH_MAX_ORDER, ARIR_FILE = 5, "res/ARIR/DRIR_SBS_VSA_50RS_SBC.sofa"
# # SH_MAX_ORDER, ARIR_FILE = 7, "res/ARIR/DRIR_CR1_VSA_86RS_L.sofa"
# # SH_MAX_ORDER, ARIR_FILE = 7, "res/ARIR/DRIR_CR1_VSA_86RS_R.sofa"
# # SH_MAX_ORDER, ARIR_FILE = 7, "res/ARIR/DRIR_CR7_VSA_86RS_L.sofa"
# # SH_MAX_ORDER, ARIR_FILE = 7, "res/ARIR/DRIR_CR7_VSA_86RS_R.sofa"
# # SH_MAX_ORDER, ARIR_FILE = 7, "res/ARIR/DRIR_LBS_VSA_86RS_PAC.sofa"
# # SH_MAX_ORDER, ARIR_FILE = 7, "res/ARIR/DRIR_LBS_VSA_86RS_PAL.sofa"
# # SH_MAX_ORDER, ARIR_FILE = 7, "res/ARIR/DRIR_LBS_VSA_86RS_SBC.sofa"
# # SH_MAX_ORDER, ARIR_FILE = 7, "res/ARIR/DRIR_LBS_VSA_86RS_SBL.sofa"
# # SH_MAX_ORDER, ARIR_FILE = 7, "res/ARIR/DRIR_SBS_VSA_86RS_PAC.sofa"  # example
# # SH_MAX_ORDER, ARIR_FILE = 7, "res/ARIR/DRIR_SBS_VSA_86RS_PAL.sofa"
# # SH_MAX_ORDER, ARIR_FILE = 7, "res/ARIR/DRIR_SBS_VSA_86RS_PAR.sofa"
# # SH_MAX_ORDER, ARIR_FILE = 7, "res/ARIR/DRIR_SBS_VSA_86RS_SBC.sofa"
# # SH_MAX_ORDER, ARIR_FILE = 8, "res/ARIR/DRIR_CR1_VSA_110RS_L.sofa"  # example
# # SH_MAX_ORDER, ARIR_FILE = 8, "res/ARIR/DRIR_CR1_VSA_110RS_R.sofa"
# # SH_MAX_ORDER, ARIR_FILE = 8, "res/ARIR/DRIR_CR7_VSA_110RS_L.sofa"
# # SH_MAX_ORDER, ARIR_FILE = 8, "res/ARIR/DRIR_CR7_VSA_110RS_R.sofa"
# # SH_MAX_ORDER, ARIR_FILE = 8, "res/ARIR/DRIR_LBS_VSA_110RS_PAC.sofa"
# # SH_MAX_ORDER, ARIR_FILE = 8, "res/ARIR/DRIR_LBS_VSA_110RS_PAL.sofa"
# # SH_MAX_ORDER, ARIR_FILE = 8, "res/ARIR/DRIR_LBS_VSA_110RS_SBC.sofa"
# # SH_MAX_ORDER, ARIR_FILE = 8, "res/ARIR/DRIR_LBS_VSA_110RS_SBL.sofa"
# # SH_MAX_ORDER, ARIR_FILE = 8, "res/ARIR/DRIR_SBS_VSA_110RS_PAC.sofa"
# # SH_MAX_ORDER, ARIR_FILE = 8, "res/ARIR/DRIR_SBS_VSA_110RS_PAL.sofa"
# # SH_MAX_ORDER, ARIR_FILE = 8, "res/ARIR/DRIR_SBS_VSA_110RS_PAR.sofa"
# # SH_MAX_ORDER, ARIR_FILE = 8, "res/ARIR/DRIR_SBS_VSA_110RS_SBC.sofa"
# # SH_MAX_ORDER, ARIR_FILE = 8, "res/ARIR/DRIR_CR1_VSA_110OSC_L.sofa"
# # SH_MAX_ORDER, ARIR_FILE = 8, "res/ARIR/DRIR_CR1_VSA_110OSC_R.sofa"
# # SH_MAX_ORDER, ARIR_FILE = 9, "res/ARIR/DRIR_CR7_VSA_146OSC_L.sofa"
# # SH_MAX_ORDER, ARIR_FILE = 9, "res/ARIR/DRIR_CR7_VSA_146OSC_R.sofa"
# SH_MAX_ORDER, ARIR_FILE = 11, "res/ARIR/DRIR_LBS_VSA_194OSC_PAC.sofa"  # example
# # SH_MAX_ORDER, ARIR_FILE = 11, "res/ARIR/DRIR_LBS_VSA_194OSC_PAL.sofa"
# # SH_MAX_ORDER, ARIR_FILE = 11, "res/ARIR/DRIR_LBS_VSA_194OSC_SBC.sofa"
# # SH_MAX_ORDER, ARIR_FILE = 11, "res/ARIR/DRIR_LBS_VSA_194OSC_SBL.sofa"
# # SH_MAX_ORDER, ARIR_FILE = 11, "res/ARIR/DRIR_SBS_VSA_194OSC_PAC.sofa"
# # SH_MAX_ORDER, ARIR_FILE = 11, "res/ARIR/DRIR_SBS_VSA_194OSC_PAL.sofa"
# # SH_MAX_ORDER, ARIR_FILE = 11, "res/ARIR/DRIR_SBS_VSA_194OSC_PAR.sofa"
# # SH_MAX_ORDER, ARIR_FILE = 11, "res/ARIR/DRIR_SBS_VSA_194OSC_SBC.sofa"
# # SH_MAX_ORDER, ARIR_FILE = 12, "res/ARIR/DRIR_CR1_VSA_1202RS_L.sofa"
# # SH_MAX_ORDER, ARIR_FILE = 12, "res/ARIR/DRIR_CR1_VSA_1202RS_R.sofa"
# # SH_MAX_ORDER, ARIR_FILE = 12, "res/ARIR/DRIR_CR7_VSA_1202RS_L.sofa"
# # SH_MAX_ORDER, ARIR_FILE = 12, "res/ARIR/DRIR_CR7_VSA_1202RS_R.sofa"
# # SH_MAX_ORDER, ARIR_FILE = 12, "res/ARIR/DRIR_LBS_VSA_1202RS_PAC.sofa"  # example
# # SH_MAX_ORDER, ARIR_FILE = 12, "res/ARIR/DRIR_LBS_VSA_1202RS_PAL.sofa"
# # SH_MAX_ORDER, ARIR_FILE = 12, "res/ARIR/DRIR_LBS_VSA_1202RS_SBC.sofa"
# # SH_MAX_ORDER, ARIR_FILE = 12, "res/ARIR/DRIR_LBS_VSA_1202RS_SBL.sofa"
# # SH_MAX_ORDER, ARIR_FILE = 12, "res/ARIR/DRIR_SBS_VSA_1202RS_PAC.sofa"
# # SH_MAX_ORDER, ARIR_FILE = 12, "res/ARIR/DRIR_SBS_VSA_1202RS_PAL.sofa"
# # SH_MAX_ORDER, ARIR_FILE = 12, "res/ARIR/DRIR_SBS_VSA_1202RS_PAR.sofa"
# # SH_MAX_ORDER, ARIR_FILE = 12, "res/ARIR/DRIR_SBS_VSA_1202RS_SBC.sofa"

# ############ BRIR RENDERING OF SOUND_FIELD_ANALYSIS OUTPUT ############
# SOURCE_FILE = "res/source/Drums_48.wav"
# TRACKER_TYPE = "AUTO_ROTATE"  # overwrite in case you want to use head-tracking
# ARIR_TYPE = None
# HRIR_TYPE = "BRIR_SSR"
# HRIR_FILE = "res/HRIR/KU100_THK/BRIR_CR1_VSA_110RS_L_SSR_SFA_-37_SOFA_RFI.wav"
# HRIR_LEVEL = -12

# ############ HRIR RENDERING OF ARBITRARY VIRTUAL SOURCE NUMBERS ############
# SOURCE_FILE = "res/source/PinkMartini_Lilly_44.wav"
# SOURCE_POSITIONS = [(30, 0), (-30, 0)]
# """[in case related to ARIR] reference frontal position as list of tuple of azimuth
# (counterclockwise) and elevation in degrees (int or float)."""
# TRACKER_TYPE = "AUTO_ROTATE"  # overwrite in case you want to use head-tracking
# ARIR_TYPE = None
# HRIR_TYPE = "HRIR_SSR"
# HRIR_FILE = "res/HRIR/FABIAN_TUB/hrirs_fabian.wav"
# HRIR_LEVEL = 0


# ################## #
#  LOGGING SETTINGS  #
# ################## #

# LOGGING_LEVEL = "DEBUG"
LOGGING_LEVEL = "INFO"
# LOGGING_LEVEL = "WARNING"
"""Lowest level of messages being printed to the logs, see `process_logger`."""

LOGGING_FORMAT = "%(name)-@s  %(levelname)-8s  %(message)s"
"""Format of messages being printed to the log, see `process_logger`."""

LOGGING_PATH = "log/"
"""Path of log messages being saved to, see `process_logger`."""


# ###################### #
#  PERFORMANCE SETTINGS  #
# ###################### #

# PROCESS_PRIORITY = 1
# """Process attribute, which is requested as its operation system specific priority,
# see `__main__`. """

IS_SINGLE_PRECISION = True
"""If all signal generation and processing should be done in single precision (`numpy32` and
`complex64`) instead of double precision (`float64` and `complex128`). """

PROCESS_FILE_LIMIT_MIN = 10240
"""MacOS specific process attribute to limit the number of open files, which is requested to be
as high as the specified value. """

NUMPY_OMP_NUM_THREADS = 1  # 1 leads to best performance so far
"""Number of available OpenMP threads, should be chosen carefully to not lead to oversubscription
(1 is equivalent to disabling threading in OpenMP). """
NUMPY_OMP_DYNAMIC = False  # False leads to best performance so far
"""Enable dynamic reduction of number of threads based on analysis of system workload, which may
reduce possible oversubscription from OpenMP threading. """
# quarter CPU count leads to best real-time performance so far
NUMPY_MKL_NUM_THREADS = int(tools.get_cpu_count() / 4)
"""Number of available MKL threads, should be chosen carefully to not lead to oversubscription (1
is equivalent to disabling threading in MKL). """
NUMPY_MKL_DYNAMIC = False  # False leads to best performance so far
"""Enable dynamic reduction of number of threads based on analysis of system workload, which may
reduce possible oversubscription from MKL threading. """

IS_PYFFTW_MODE = True  # True leads to best performance so far
"""If `pyfftw` package (wrapper for FFTW library) should be used instead of `numpy` for all
real-time DFT operations. In case `pyfftw` is not used, all related tasks like loading/saving and
pre-calculating FFTW wisdom will be skipped. """
# PYFFTW_EFFORT = "FFTW_ESTIMATE"
# PYFFTW_EFFORT = "FFTW_MEASURE"
PYFFTW_EFFORT = "FFTW_PATIENT"
# PYFFTW_EFFORT = "FFTW_EXHAUSTIVE"
"""Amount of effort spent during the FFTW planning stage to create the fastest possible
transform, see `pyfftw`. """
# quarter CPU count leads to best real-time performance so far
PYFFTW_NUM_THREADS = int(tools.get_cpu_count() / 4)
"""Number of available FFTW threads, should be chosen carefully to not lead to oversubscription (
1 is equivalent to disabling threading in FFTW). """
PYFFTW_WISDOM_FILE = f"{LOGGING_PATH}pyfftw_wisdom_{tools.get_system_name()}.bin"
"""File of `pyfftw` wisdom being loaded/saved by the application, see `__main__`."""
PYFFTW_LEGACY_FILE = None
# PYFFTW_LEGACY_FILE = f"{LOGGING_PATH}pyfftw_wisdom.bin"
"""File of `pyfftw` wisdom being loaded by the application without signature validation,
see `__main__`."""


# ################# #
#  ! DO NOT EDIT !  #
# ################# #

IS_DEBUG_MODE = tools.get_is_debug()
"""If the application is run in a debugging mode. When execution is paused this is used so
certain processes relying on real time execution do not raise errors. Also this is used to make
use of breakpoints in certain functions before they get released in a separate process. """

IS_RUNNING = mp_context.Event()
"""If the application is running and rendering audio at the moment. This needs to be set after
all rendering clients have started up. This can also be used to globally interrupt rendering and
output of all clients. """

# fmt: on

if "STUDY_MODE" not in locals():
    STUDY_MODE = False
    """Run regular rendering mode with enforced config to be used in user studies, see `tools`
    command line parsing. """
if "BENCHMARK_MODE" not in locals():
    BENCHMARK_MODE = False
    """Run mode specifying a benchmarking method, see `tools` command line parsing."""
if "VALIDATION_MODE" not in locals():
    VALIDATION_MODE = False
    """Run mode specifying a validation method against provided reference impulse response set,
    see `tools` command line parsing. """
if "DEVELOPER_MODE" not in locals():
    DEVELOPER_MODE = False
    """Run mode specifying a development test method, see `tools` command line parsing."""

if "BLOCK_LENGTH" not in locals():
    BLOCK_LENGTH = None
if "IR_TRUNCATION_LEVEL" not in locals():
    IR_TRUNCATION_LEVEL = 0
if "SH_MAX_ORDER" not in locals():
    SH_MAX_ORDER = None
if "SH_COMPENSATION_TYPE" not in locals():
    SH_COMPENSATION_TYPE = None
if "SH_IS_ENFORCE_PINV" not in locals():
    SH_IS_ENFORCE_PINV = False
if "ARIR_RADIAL_AMP" not in locals():
    ARIR_RADIAL_AMP = 0
if "SOURCE_POSITIONS" not in locals():
    SOURCE_POSITIONS = [(0, 0)]
if "TRACKER_TYPE" not in locals():
    TRACKER_TYPE = None
if "TRACKER_PORT" not in locals():
    TRACKER_PORT = None
if "REMOTE_OSC_PORT" not in locals():
    REMOTE_OSC_PORT = None
if "G_TYPE" not in locals():
    G_TYPE = None
if "ARIR_TYPE" not in locals():
    ARIR_TYPE = None
if "HPCF_TYPE" not in locals():
    HPCF_TYPE = None
if "SOURCE_LEVEL" not in locals():
    SOURCE_LEVEL = 0
if "G_LEVEL" not in locals():
    G_LEVEL = 0
if "ARIR_LEVEL" not in locals():
    ARIR_LEVEL = 0
if "HRIR_LEVEL" not in locals():
    HRIR_LEVEL = 0
if "HPCF_LEVEL" not in locals():
    HPCF_LEVEL = 0
if "G_LEVEL_REL" not in locals():
    G_LEVEL_REL = 0
if "SOURCE_MUTE" not in locals():
    SOURCE_MUTE = False
if "G_MUTE" not in locals():
    G_MUTE = False
if "ARIR_MUTE" not in locals():
    ARIR_MUTE = False
if "HRIR_MUTE" not in locals():
    HRIR_MUTE = False
if "HPCF_MUTE" not in locals():
    HPCF_MUTE = False
if "HRIR_DELAY" not in locals():
    HRIR_DELAY = 0
if "LOGGING_PATH" not in locals():
    LOGGING_PATH = None

# transform following attributes from relative package into absolute system paths
HRIR_FILE = tools.get_absolute_from_relative_package_path(HRIR_FILE)
# noinspection PyUnboundLocalVariable,PyUnresolvedReferences
ARIR_FILE = (
    tools.get_absolute_from_relative_package_path(ARIR_FILE)
    if "ARIR_FILE" in locals()
    else None
)
# noinspection PyUnboundLocalVariable,PyUnresolvedReferences
SOURCE_FILE = (
    tools.get_absolute_from_relative_package_path(SOURCE_FILE)
    if "SOURCE_FILE" in locals()
    else None
)
# noinspection PyUnboundLocalVariable,PyUnresolvedReferences
HPCF_FILE = (
    tools.get_absolute_from_relative_package_path(HPCF_FILE)
    if "HPCF_FILE" in locals()
    else None
)
# noinspection PyUnboundLocalVariable,PyUnresolvedReferences

# manually call to update logging path
LOGGING_PATH = tools.get_absolute_from_relative_package_path(LOGGING_PATH)
