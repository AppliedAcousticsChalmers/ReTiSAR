import logging
import os
import sys
from time import sleep

if sys.platform == 'darwin':
    # prevent exception due to python not being a framework build when installed
    import matplotlib

    matplotlib.use("TkAgg")
    del matplotlib
    import matplotlib.pyplot as plt
else:
    import matplotlib.pyplot as plt

# reset matplotlib logging level
logging.getLogger('matplotlib').setLevel(logging.INFO)

SPEED_OF_SOUND = 343
"""Speed of sound in meters per second in air."""
SEPARATOR = '-------------------------------------------------------------------------'
"""String to improve visual orientation for a clear logging behaviour."""


def parse_cmd_args():
    """Allow for parsing of certain command line arguments and update according values in `config`."""
    import argparse
    from . import config

    # create parser and introduce all possible arguments
    parser = argparse.ArgumentParser(description='Implementation of a real-time binaural spherical microphone '
                                                 'renderer in Python.')
    parser.add_argument('-b', '--BLOCK_LENGTH', type=int, required=False,
                        help='block length of the JACK audio server and clients in samples')
    parser.add_argument('-sh', '--SH_MAX_ORDER', type=int, required=False,
                        help='spherical harmonics order when rendering Array Room Impulse Responses')
    parser.add_argument('-s', '--SOURCE_FILE', type=str, required=False,
                        help='file of audio being played by the application')
    parser.add_argument('-sp', '--SOURCE_POSITIONS', type=str, required=False,
                        help='source positions as list of tuple of azimuth and elevation in degrees')
    parser.add_argument('-sl', '--SOURCE_LEVEL', type=int, required=False,
                        help='output level of source audio replay')
    parser.add_argument('-sm', '--SOURCE_MUTE', type=transform_str2bool, required=False,
                        help='output mute state of source audio replay')
    parser.add_argument('-gt', '--G_TYPE', type=str, required=False,
                        choices=['NOISE_WHITE', 'NOISE_IIR_PINK', 'NOISE_AR_PINK', 'NOISE_AR_PURPLE', 'NOISE_AR_BLUE',
                                 'NOISE_AR_BROWN'],
                        help='type of algorithm used by generator to create sound')
    parser.add_argument('-gl', '--G_LEVEL', type=int, required=False,
                        help='output level of sound generator')
    parser.add_argument('-gm', '--G_MUTE', type=transform_str2bool, required=False,
                        help='output mute state of sound generator')
    parser.add_argument('-ar', '--ARIR_FILE', type=str, required=False,
                        help='file with FIR filter containing Array Room Impulse Responses')
    parser.add_argument('-art', '--ARIR_TYPE', type=str, required=False,
                        choices=['ARIR_MIRO', 'AS_MIRO'],
                        help='type of FIR filter file containing Array Room Impulse Responses / stream configuration')
    parser.add_argument('-arl', '--ARIR_LEVEL', type=int, required=False,
                        help='output level of renderer for Array Room Impulse Response')
    parser.add_argument('-arm', '--ARIR_MUTE', type=transform_str2bool, required=False,
                        help='output mute state of renderer for Array Room Impulse Response')
    parser.add_argument('-arr', '--ARIR_RADIAL_AMP', type=int, required=False,
                        help='maximum amplification limit in dB when generating modal radial filters')
    parser.add_argument('-hr', '--HRIR_FILE', type=str, required=False,
                        help='file with FIR filter containing Head Related Impulse Responses')
    parser.add_argument('-hrt', '--HRIR_TYPE', type=str, required=False,
                        choices=['HRIR_SSR', 'BRIR_SSR', 'HRIR_MIRO'],
                        help='type of FIR filter file containing Head Related Impulse Responses')
    parser.add_argument('-hrl', '--HRIR_LEVEL', type=int, required=False,
                        help='output level of renderer for Head Related Impulse Response')
    parser.add_argument('-hrm', '--HRIR_MUTE', type=transform_str2bool, required=False,
                        help='output mute state of renderer for Head Related Impulse Response')
    parser.add_argument('-hp', '--HPIR_FILE', type=str, required=False,
                        help='file with FIR filter containing Headphone Equalization Impulse Responses')
    # parser.add_argument('-hpt', '--HPIR_TYPE', type=str, required=False,
    #                     choices=['FIR_MULTICHANNEL'],
    #                     help='type of FIR filter file containing Headphone Equalization Impulse Responses')
    parser.add_argument('-hpl', '--HPIR_LEVEL', type=int, required=False,
                        help='output level of renderer for Headphone Equalization Impulse Response')
    parser.add_argument('-hpm', '--HPIR_MUTE', type=transform_str2bool, required=False,
                        help='output mute state of renderer for Headphone Equalization Impulse Response')
    parser.add_argument('-t', '--TRACKER_PORT', type=str, required=False,
                        help='system specific path to tracker port to read data from')
    parser.add_argument('-tt', '--TRACKER_TYPE', type=str, required=False,
                        choices=['NONE', 'AUTO_ROTATE', 'POLHEMUS_PATRIOT', 'RAZOR_AHRS'],
                        help='type information of hardware providing head tracking data')
    parser.add_argument('-r', '--REMOTE_OSC_PORT', type=int, required=False,
                        help='port to receive Open Sound Control remote messages')
    parser.add_argument('-ll', '--LOGGING_LEVEL', type=str, required=False,
                        choices=['NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='lowest logging level being shown and printed to the logs')
    parser.add_argument('-lp', '--LOGGING_PATH', type=str, required=False,
                        help='path of log messages being saved to')
    parser.add_argument('-bm', '--BENCHMARK_MODE', type=str, required=False,
                        choices=['PARALLEL_CLIENTS', 'PARALLEL_CONVOLVERS'],
                        help='run benchmark mode with specified method, ignoring all other parameters')
    parser.add_argument('-vm', '--VALIDATION_MODE', type=str, required=False,
                        help='run validation mode against provided reference impulse response set')
    parser.add_argument('-dm', '--DEVELOPER_MODE', action='store_true', required=False,
                        help='run development test mode')

    # parse arguments
    if len(sys.argv) > 1:
        print('parsing command line arguments ...')
    else:
        print('no command line arguments given (see `-h` or `--help`` for instructions).')
    args = parser.parse_args()

    # update config
    for a in args.__dict__:
        value = args.__dict__[a]
        if value is not None:
            try:
                value_default = getattr(config, a)

                # if parameter is path, transform into relative path
                if value_default is not None and not isinstance(value_default, list) and os.path.isfile(value_default):
                    value_default_rel = os.path.relpath(value_default)
                else:
                    value_default_rel = value_default

                # if parameter is path, transform into absolute path
                if value is not None and os.path.isfile(value):
                    value = os.path.abspath(value)
                    value_rel = os.path.relpath(value)
                else:
                    # if value is list
                    if isinstance(value_default, list):
                        if not ('[(' in value and ',' in value and ')]' in value):
                            print('config.{:<20} has incorrect shape.\n'
                                  'application interrupted.'.format(a), file=sys.stderr)
                            sys.exit(1)

                        import ast
                        value = ast.literal_eval(value)

                    value_rel = value

                if value != value_default:
                    # set parameter in config
                    print('config.{:<20} using {:<40} (default "{}").'.format(
                        a, '"' + str(value_rel) + '"', value_default_rel))
                    setattr(config, a, value)

            except AttributeError:
                print('config.{}   parameter unknown.\n'
                      'application interrupted.'.format(a), file=sys.stderr)
                sys.exit(1)
    print('all unnamed arguments use default values (see module `config`).')


def get_is_debug():
    """
    The current implementation works fine in PyCharm, but might not work from command line or other IDEs.

    Returns
    -------
    bool
        if the application is run in a debugging mode
    """
    return sys.gettrace() is not None


def get_cpu_count():
    """
    Returns
    -------
    int
        number of system CPU cores available reported by `os`
    """
    # import multiprocessing
    # return multiprocessing.cpu_count()
    return os.cpu_count()


def get_absolute_from_relative_package_path(relative_package_path):
    """
    Parameters
    ----------
    relative_package_path : str or list of str
        path to a resource relative to the package base directory

    Returns
    -------
    str or list of str
        absolute system path to resource
    """
    _PACKAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))

    if not relative_package_path\
            or (type(relative_package_path) == str and relative_package_path.strip('\'"') == ''):
        return None
    elif type(relative_package_path) == str:
        return os.path.join(_PACKAGE_DIR, relative_package_path)
    else:  # list
        absolute_paths = []
        for p in relative_package_path:
            absolute_paths.append(os.path.join(_PACKAGE_DIR, p))
        return absolute_paths


def request_process_parameters():
    """
    Set session specific limitations for the current process like number of processes and open files to its maximum
    possible value. If the individual limit could not be set only its current value is printed.
    """
    import resource
    from . import config

    print('trying to adjust process parameters ...')

    if hasattr(config, 'PROCESS_PRIORITY') and config.PROCESS_PRIORITY != 0:
        print('[WARNING]  setting of process priority currently not implemented.', file=sys.stderr)
        # import psutil
        # sleep(.2)  # to get correct output order
        # p = psutil.Process(os.getpid())
        # nice = p.nice()
        # try:
        #     p.nice(-config.PROCESS_PRIORITY)  # negative value of NICE on OSX !!!
        #     nice_new = p.nice()
        #     if nice == nice_new:
        #         raise PermissionError()
        #     print('set process priority (OSX nice, lower values mean higher priority) from {} to {}.'.format(
        #         nice, nice_new))
        # except psutil.AccessDenied:
        #     print('[WARNING]  process priority could not be set over {}.\n'
        #           ' --> Run Python with `sudo` for elevated permissions!'.format(nice), file=sys.stderr)
        # sleep(.2)  # to get correct output order

    lim = resource.getrlimit(resource.RLIMIT_NPROC)
    try:
        resource.setrlimit(resource.RLIMIT_NPROC, (lim[1], lim[1]))
        lim_new = resource.getrlimit(resource.RLIMIT_NPROC)
        if lim[0] == lim_new[0]:
            raise ValueError()
        print('set maximum number of processes the current process may create from {} to {}.'.format(
            lim[0], lim_new[0]))
    except ValueError:
        print('maximum number of processes the current process may create is {}.'.format(lim[0]))

    lim = resource.getrlimit(resource.RLIMIT_NOFILE)
    try:
        if not hasattr(config, 'PROCESS_FILE_LIMIT_MIN') or config.PROCESS_FILE_LIMIT_MIN < lim[0]:
            raise ValueError()
        resource.setrlimit(resource.RLIMIT_NOFILE, (config.PROCESS_FILE_LIMIT_MIN, config.PROCESS_FILE_LIMIT_MIN))
        lim_new = resource.getrlimit(resource.RLIMIT_NOFILE)
        if lim[0] == lim_new[0]:
            raise ValueError()
        print('set maximum number of open file descriptors for the current process from {} to {}.'.format(
            lim[0], lim_new[0]))
    except ValueError:
        print('maximum number of open file descriptors for the current process is {}.'.format(lim[0]))


def request_numpy_parameters():
    """
    Set `numpy` specific parameters for linked libraries like settings for automatic threading behaviour. Information
    about the utilized threading libraries is printed out afterwards.
    """

    def set_env_parameter(param, val):
        print('setting environment parameter {} from {} to {}.'.format(param, os.environ.get(param), val))
        os.environ[param] = val

    from . import config

    print('trying to adjust numpy parameters ...')

    # these variables need to be set before `numpy` is imported the first time
    set_env_parameter('OMP_DYNAMIC', config.NUMPY_OMP_DYNAMIC.__str__())
    set_env_parameter('OMP_NUM_THREADS', config.NUMPY_OMP_NUM_THREADS.__str__())
    set_env_parameter('MKL_DYNAMIC', config.NUMPY_MKL_DYNAMIC.__str__())
    set_env_parameter('MKL_NUM_THREADS', config.NUMPY_MKL_NUM_THREADS.__str__())

    # set_env_parameter('OMP_NESTED', 'TRUE')  # no positive effect on performance
    # set_env_parameter('NUMEXPR_NUM_THREADS', '1')  # no positive effect on performance
    # set_env_parameter('OPENBLAS_NUM_THREADS', '1')  # no positive effect on performance
    # set_env_parameter('VECLIB_MAXIMUM_THREADS', '1')  # no positive effect on performance
    # set_env_parameter('MKL_DOMAIN_NUM_THREADS', '"MKL_FFT=1"')  # no positive effect on performance

    print_numpy_info()

    # show shape when printing  `np.ndarray` (useful while debugging)
    import numpy as np
    np.set_string_function(lambda ndarray: '[{}{}{}] {} {}'.format(['x', 'C'][ndarray.flags.carray],
                                                                   ['x', 'F'][ndarray.flags.farray],
                                                                   ['x', 'O'][ndarray.flags.owndata],
                                                                   ndarray.dtype, ndarray.shape), repr=False)


def print_numpy_info():
    """
    Prints out numpy information about how `numpy` is linked by checking what symbols are defined when loading the
    numpy modules using BLAS.

    Source: https://gist.github.com/seberg/ce4563f3cb00e33997e4f80675b80953
    """
    import ctypes
    import numpy as np

    try:
        # noinspection PyProtectedMember
        multiarray = np.core._multiarray_umath.__file__
    except AttributeError:
        multiarray = np.core.multiarray.__file__
    dll = ctypes.CDLL(multiarray)

    blas = []
    implementations = {'openblas_get_num_threads': 'openblas',
                       'ATL_buildinfo': 'atlas',
                       'bli_thread_get_num_threads': 'blis',
                       'MKL_Get_Max_Threads': 'MKL', }

    for func, implementation in implementations.items():
        try:
            getattr(dll, func)
            blas.append(implementation)
        except AttributeError:
            continue

    if len(blas) > 1:
        print('[WARNING]  multiple BLAS/LAPACK libs loaded: {}'.format(blas))

    if len(blas) == 0:
        print('[WARNING]  unable to guess BLAS implementation, it is not one of: {}'.format(implementations.values()))
        print(' --> additional symbols are not loaded?!')

    link_str = 'numpy linked to'
    for impl in blas:
        if impl == 'openblas':
            dll.openblas_get_config.restype = ctypes.c_char_p
            dll.openblas_get_num_threads.restype = ctypes.c_int
            print('{} "{}" (num threads: {})\n --> {}'.format(
                link_str, 'OpenBLAS', dll.openblas_get_num_threads(), dll.openblas_get_config().decode('utf8').strip()))

        elif impl == 'blis':
            dll.bli_thread_get_num_threads.restype = ctypes.c_int
            print('{} "{}" (num threads: {}, threads enabled: {}').format(
                link_str, 'BLIS', dll.bli_thread_get_num_threads(), dll.bli_info_get_enable_threading())

        elif impl == 'MKL':
            version_func = dll.mkl_get_version_string
            version_func.argtypes = (ctypes.c_char_p, ctypes.c_int)
            out_buf = ctypes.c_buffer(500)
            version_func(out_buf, 500)
            print('{} "{}" (max threads: {})\n --> {}'.format(
                link_str, 'MKL', dll.MKL_Get_Max_Threads(), out_buf.value.decode('utf8').strip()))

        elif impl == 'atlas':
            print('{} "{}" (ATLAS is thread-safe, max number of threads are fixed at compile time)\n --> {}'.format(
                link_str, 'ATLAS', dll.ATL_buildinfo()))

        else:
            print('{} "{}"'.format(link_str, impl))

    if 'MKL' not in blas:
        print('[WARNING]  "MKL" version of `numpy` is not linked, which is supposed to provide best performance.',
              file=sys.stderr)


def import_fftw_wisdom():
    """
    Load and import gathered FFTW wisdom from provided file and set global `pyfftw` parameters according to
    configuration. If no wisdom can be imported, information is given that it will be generated before audio
    rendering starts.
    """

    def log_error(log_str):
        print(log_str, file=sys.stderr)
        print(' --> All necessary wisdom will be generated now. This might take a while, before the rendering will '
              'start.\n --> Take care to properly terminate this application to have the gathered wisdom exported!',
              file=sys.stderr)
        sleep(.05)  # to get correct output order

    import pickle
    from . import config

    print('loading gathered FFTW wisdom from "{}" ...'.format(os.path.relpath(config.PYFFTW_WISDOM_FILE)))
    try:
        # load from file
        with open(config.PYFFTW_WISDOM_FILE, 'rb') as f:
            wisdom = pickle.load(f)

        # load wisdom
        import pyfftw
        pyfftw.import_wisdom(wisdom)

        # print wisdom
        for w in wisdom:
            n = w.decode('utf-8').strip().split('\n')
            print(' --> {:>3} entries for "{}"'.format(len(n) - 2, n[0].strip('()')))

        # set global config parameters
        if not config.DEVELOPER_MODE and hasattr(config, 'PYFFTW_NUM_THREADS')\
                and pyfftw.config.NUM_THREADS != config.PYFFTW_NUM_THREADS:
            print('setting `pyfftw` environment parameter {} from {} to {}.'.format(
                'NUM_THREADS', pyfftw.config.NUM_THREADS, config.PYFFTW_NUM_THREADS))
            pyfftw.config.NUM_THREADS = config.PYFFTW_NUM_THREADS
        else:
            print('`pyfftw` environment parameter {} is {}.'.format('NUM_THREADS', pyfftw.config.NUM_THREADS))

        if not config.DEVELOPER_MODE and hasattr(config, 'PYFFTW_EFFORT')\
                and pyfftw.config.PLANNER_EFFORT != config.PYFFTW_EFFORT:
            print('setting `pyfftw` environment parameter {} from {} to {}.'.format(
                'PLANNER_EFFORT', pyfftw.config.PLANNER_EFFORT, config.PYFFTW_EFFORT))
            pyfftw.config.PLANNER_EFFORT = config.PYFFTW_EFFORT
        else:
            print('`pyfftw` environment parameter {} is {}.'.format('PLANNER_EFFORT', pyfftw.config.PLANNER_EFFORT))

    except FileNotFoundError:
        log_error('... file not found.')
    except EOFError:
        log_error('... error reading file.')

        # rename existing file as backup
        backup = os.path.join(
            os.path.dirname(config.PYFFTW_WISDOM_FILE), 'BACKUP_' + os.path.basename(config.PYFFTW_WISDOM_FILE))
        os.rename(config.PYFFTW_WISDOM_FILE, backup)
        print('... renamed existing file to "{}".'.format(os.path.relpath(backup)), file=sys.stderr)


def export_fftw_wisdom(logger):
    """
    Write gathered FFTW wisdom to provided file for later import.

    Parameters
    ----------
    logger : logging.Logger or None
        instance to provide identical logging behaviour as the calling process
    """
    import pyfftw
    import pickle
    from . import config

    log_str = 'writing gathered FFTW wisdom to "{}" ...'.format(os.path.relpath(config.PYFFTW_WISDOM_FILE))
    logger.info(log_str) if logger else print(log_str)

    with open(config.PYFFTW_WISDOM_FILE, 'wb') as f:
        pickle.dump(pyfftw.export_wisdom(), f, protocol=pickle.HIGHEST_PROTOCOL)


def get_pretty_delay_str(samples, fs):
    """
    Parameters
    ----------
    samples : int or float or numpy.ndarray
        amount of samples
    fs : int
        sampling frequency

    Returns
    -------
    str
        generated string with printed values in absolute samples as well as according delay in milliseconds and traveled
        sound distance in meters
    """
    import numpy as np

    # calculate according time delay and distances
    samples = np.abs(samples)
    delay = samples / fs  # in seconds
    distance = delay * SPEED_OF_SOUND  # in meter

    # generate string
    return '{} samples / {} ms / {} m'.format(np.array2string(samples),
                                              np.array2string(delay * 1000, precision=1),
                                              np.array2string(distance, precision=3))
    # return '{:d} samples / {:.1f} ms / {:.3f} mm'.format(samples, delay * 1000, distance)


def transform_into_type(str_or_instance, _type):
    """
    Parameters
    ----------
    str_or_instance : str, Type or None
        string or instance of type that should be transformed
    _type : type
        type that should be transformed into

    Returns
    -------
    class
        type instance

    Raises
    ------
    ValueError
        in case unknown type is given
    """
    if str_or_instance is None:
        return None
    elif type(str_or_instance) is str:
        if str_or_instance.upper() == 'NONE':
            return None
        # transform string into enum, will fail in case an invalid type string was given
        # noinspection PyUnresolvedReferences
        return _type[str_or_instance]
    elif isinstance(str_or_instance, _type):
        return str_or_instance
    else:
        raise ValueError('unknown parameter type {}, see `{}` for reference!'.format(type(str_or_instance), _type))


def transform_str2bool(_str):
    """
    Parameters
    ----------
    _str : str
        equivalent string to be transformed into a boolean

    Returns
    -------
    bool
        boolean transformed from equivalent string

    Raises
    ------
    ValueError
        in case unknown equivalent string was given
    """
    if _str.upper() in ('TRUE', 'YES', 'T', 'Y', '1'):
        return True
    elif _str.upper() in ('FALSE', 'NO', 'F', 'N', '0'):
        return False
    elif _str.upper() in ('TOGGLE', 'SWITCH', 'T', 'S', '-1'):
        return None
    else:
        raise ValueError('unknown boolean equivalent string "{}".'.format(_str))


def transform_into_state(state, logger=None):
    """
    Parameters
    ----------
    state : bool, int, float, str or None
        state value in compatible format for which a mapping will be achieved, if an invalid value is given a warning
        will be logged and `None` returned
    logger : logging.Logger, optional
        instance to provide identical logging behaviour as the calling process

    Returns
    -------
    bool or None
        state value as either True, False or None
    """
    if state is None or type(state) is bool:
        return state

    # parse str
    if type(state) is str:
        return transform_str2bool(state.strip())

    # parse int and float
    if type(state) in [int, float]:
        state = int(state)
        if state == 1:
            return True
        if state == 0:
            return False
        if state == -1:
            return None

    # no match found
    log_str = 'unknown state "{}"'.format(state)
    logger.warning(log_str) if logger else print(log_str, file=sys.stderr)
    return None


def generate_noise(shape, scale=1 / 7, is_complex=False):
    """
    Parameters
    ----------
    shape : tuple of int
        shape of noise to generate (last axis contains normally distributed time samples)
    scale : float, optional
        numpy.random.normal scaling factor, the default value is supposed to result in amplitudes [-1, 1]
    is_complex : bool, optional
        if generated data should be complex

    Returns
    -------
    numpy.ndarray
        generated white noise (normal distributed) with given shape
    """
    import numpy as np

    normal = np.random.normal(0, scale, shape)
    if is_complex:
        return normal + 1j * np.random.normal(0, scale, shape)
    else:
        return normal


def generate_highpass_td(is_iir, fs, cutoff, iir_order=None, fir_samples=None):
    """
    Parameters
    ----------
    is_iir : bool
        if IIR filter coefficients should be generated, otherwise a minimal phase FIR filter will be generated
    fs : int
        sampling frequency
    cutoff : float
        cutoff frequency of generated highpass
    iir_order : int, optional
        filter order of generated highpass (in case of IIR filter)
    fir_samples : int, optional
        filter length of generated highpass (in case of FIR filter)

    Returns
    -------
    numpy.ndarray
        filter b coefficients
    numpy.ndarray
        filter a coefficients (None in case of FIR filter)

    Raises
    ------
    ValueError
        in case no value for IIR filter order or FIR filter length is given
    """
    from scipy import signal

    if is_iir:
        if iir_order is None:
            raise ValueError('An IIR filter order must be given.')
        # noinspection PyTupleAssignmentBalance
        b, a = signal.butter(iir_order, cutoff, output='ba', btype='highpass', analog=False, fs=fs)
        return b, a

    else:
        if fir_samples is None:
            raise ValueError('An FIR filter target length must be given.')
        # # odd must be given here
        # fir_samples = (2 * fir_samples) - 1
        # b = signal.firls(fir_samples, [0, cutoff - cutoff/6, cutoff + cutoff/6, fs/2], [0, 0, 1, 1], fs=fs)
        #
        # # length will be halved here
        # b = signal.minimum_phase(b, 'hilbert', n_fft=fs)
        #
        # # target length should match here
        # return b, None
        raise NotImplementedError('function is not properly implemented yet.')


def generate_delay_fd(length_samples, fs, delay_seconds):
    """
    Parameters
    ----------
    length_samples : int
        number of bins in frequency domain (half-sided spectrum)
    fs : int
        sampling frequency of data
    delay_seconds : float
        delay time in seconds

    Returns
    -------
    numpy.ndarray
        delay spectrum in frequency domain
    """
    import numpy as np

    freqs = np.linspace(0, fs / 2, length_samples)
    return np.exp(-1j * 2 * np.pi * freqs * delay_seconds)


def calculate_rms(data_td):
    """
    Parameters
    ----------
    data_td : numpy.ndarray
        time domain data (along last axis) the root mean square value should be calculated of

    Returns
    -------
    numpy.ndarray
        root mean square values of provided time domain data
    """
    import numpy as np

    return np.sqrt(np.mean(np.square(data_td), axis=-1))


def plot_ir_and_tf(data_td_or_fd, fs, is_share_y=False, is_label_y=False, is_td_db_y=False):
    """
    Parameters
    ----------
    data_td_or_fd : numpy.ndarray
        time (real) or frequency domain (complex) data that should be plotted
    fs : int
        sampling frequency of data
    is_share_y : bool, optional
        if y-axis dimensions of plots for all data channels should be shared
    is_label_y : bool, optional
        if y-axis of last plot should have a label
    is_td_db_y : bool, optional
        if time domain plot y-axis should be in dB_FS

    Returns
    -------
    matplotlib.figure.Figure
        generated plot
    """
    import numpy as np

    data_td_or_fd = np.atleast_2d(data_td_or_fd)
    if np.iscomplex(data_td_or_fd).any():
        # fd data given
        data_fd = data_td_or_fd
        data_td = np.fft.irfft(data_td_or_fd)
    else:
        # td data given
        data_td = data_td_or_fd
        data_fd = np.fft.rfft(data_td_or_fd)
    del data_td_or_fd

    # prevent zeros
    data_fd[np.nonzero(data_fd == 0)] = np.nan
    if is_td_db_y:
        data_td[np.nonzero(data_td == 0)] = np.nan

        # transform td data into logarithmic scale
        data_td = 20 * np.log10(np.abs(data_td))

    fig, axes = plt.subplots(nrows=data_td.shape[0], ncols=2, squeeze=False,
                             sharex='col', sharey='col' if is_share_y else False)
    for ch in range(data_td.shape[0]):
        # plot IR
        axes[ch, 0].plot(np.arange(0, len(data_td[ch])), data_td[ch], linewidth=.5, color='C0')
        axes[ch, 0].set_xlim(-1, len(data_td[ch]) + 1)
        if is_label_y and ch == data_td.shape[0] - 1:
            axes[ch, 0].set_xlabel('Samples')
        axes[ch, 0].tick_params(direction='in', top=True, bottom=True, left=True, right=True)

        # plot spectrum
        axes[ch, 1].semilogx(np.linspace(0, fs / 2, len(data_fd[ch])), 20 * np.log10(np.abs(data_fd[ch])), color='C1')
        axes[ch, 1].set_xlim(20, fs / 2)
        if is_label_y and ch == data_td.shape[0] - 1:
            axes[ch, 1].set_xlabel('Frequency')
        axes[ch, 1].tick_params(direction='in', top=True, bottom=True, left=True, right=True)

    # remove layout margins
    fig.tight_layout(pad=0)

    return fig


def export_plot(figure, name, logger=None, file_type='png'):
    """
    Parameters
    ----------
    figure : matplotlib.figure.Figure
        plot that should be exported
    name : str
        name or path of image file being exported, in case no path is given standard logging directory will be used
    logger : logging.Logger, optional
        instance to provide identical logging behaviour as the calling process
    file_type : str, optional
        image file type should be exported, multiple in the form e.g. 'png,pdf' can be used
    """
    import re

    # store into logging directory if no path is given
    if os.path.sep not in os.path.relpath(name):
        from . import config
        name = os.path.join(config.LOGGING_PATH, name)
    # store all requested file types
    for ending in re.split(r'[,.;:|/\-+]+', file_type):
        file = name + os.path.extsep + ending
        log_str = 'writing results to "{}" ...'.format(os.path.relpath(file))
        logger.info(log_str) if logger else print(log_str)
        figure.savefig(file, dpi=300)


def export_html(file_name, html_content, title=None):
    """
    Parameters
    ----------
    file_name : str
        path and name of the HTML file being written
    html_content : str
        formatted HTML string generated by something like `pandas.DataFrame.to_html()` or
        `pandas.io.formats.style.Styler.render()`
    title : str, optional
        headline being added at the beginning of the HTML body
    """
    # add header
    html_all = '''
<html>
<head>
<style>
    h2 {
        text-align: center;
        font-family: Helvetica, Arial, sans-serif;
    }
    table { 
        margin-left: auto;
        margin-right: auto;
    }
    table, th, td {
        border: 1px solid black;
        border-collapse: collapse;
    }
    th, td {
        padding: 5px;
        text-align: right;
        font-family: Helvetica, Arial, sans-serif;
    }
    table thead tr {
        text-align: center;
        font-size: 70%;
    }
    table tbody tr:hover {
        background-color: #dddddd;
    }
</style>
</head>
<body>
    '''

    # add title
    if title:
        html_all += '<h2> {} </h2>\n'.format(title)

    # add content
    html_all += html_content

    # add footer
    html_all += '''
</body>
</html>
'''

    # write file
    with open(file_name, 'w') as f:
        f.write(html_all)
