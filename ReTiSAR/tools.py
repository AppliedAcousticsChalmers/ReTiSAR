import logging
import os
import sys
from time import sleep

if sys.platform == 'darwin':
    # prevent exception due to python not being a framework build when installed
    import matplotlib  # chosen by default not non-interactive backend 'agg' (matplotlib from conda)
    # matplotlib.use('TkAgg')  # this backend lead to complete system crashes recently (matplotlib=3.1.0)
    matplotlib.use('MacOSX')  # this backend seems to work fine
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

    class _LicenseAction(argparse.Action):
        def __call__(self, _parser, namespace, values, option_string=None):
            print(open(get_absolute_from_relative_package_path('LICENSE'), mode='r', encoding='utf-8').read())
            _parser.exit()

    # class _PrecisionAction(argparse.Action):
    #     def __call__(self, _parser, namespace, values, option_string=None):
    #         config.IS_SINGLE_PRECISION = self.dest == 'SINGLE_PRECISION'  # False if self.dest == 'DOUBLE_PRECISION'

    # create parser and introduce all possible arguments
    parser = argparse.ArgumentParser(prog=__package__,
                                     description='Implementation of the Real-Time Spherical Microphone Renderer for '
                                                 'binaural reproduction in Python.')
    parser.add_argument('-l', '--license', action=_LicenseAction, nargs=0, help='show LICENSE information and exit')
    parser.add_argument('-b', '--BLOCK_LENGTH', type=int, required=False,
                        help='block length of the JACK audio server and clients in samples')
    parser.add_argument('-irt', '--IR_TRUNCATION_LEVEL', type=float, required=False,
                        help='level to individually truncate any impulse response set after load to save performance')
    parser.add_argument('-sh', '--SH_MAX_ORDER', type=int, required=False,
                        help='spherical harmonics order when rendering Array Room Impulse Responses')
    parser.add_argument('-sht', '--SH_COMPENSATION_TYPE', type=str, required=False,
                        help='type of spherical harmonics processing compensation technique, see documentation for '
                             'valid choices')
    parser.add_argument('-s', '--SOURCE_FILE', type=str, required=False,
                        help='file of audio being played by the application')
    parser.add_argument('-sp', '--SOURCE_POSITIONS', type=str, required=False,
                        help='source positions as list of tuple of azimuth and elevation in degrees')
    parser.add_argument('-sl', '--SOURCE_LEVEL', type=float, required=False,
                        help='output level of source audio replay')
    parser.add_argument('-sm', '--SOURCE_MUTE', type=transform_str2bool, required=False,
                        help='output mute state of source audio replay')
    parser.add_argument('-gt', '--G_TYPE', type=str, required=False,
                        choices=['NOISE_WHITE', 'NOISE_IIR_PINK', 'NOISE_AR_PINK', 'NOISE_AR_PURPLE', 'NOISE_AR_BLUE',
                                 'NOISE_AR_BROWN', 'NONE'],
                        help='type of algorithm used by generator to create sound')
    parser.add_argument('-gl', '--G_LEVEL', type=float, required=False,
                        help='output level of sound generator')
    parser.add_argument('-gm', '--G_MUTE', type=transform_str2bool, required=False,
                        help='output mute state of sound generator')
    parser.add_argument('-grp', '--G_REPLACE_PORT', type=int, required=False,
                        help='port ID (one channel) that will be replaced with an individual sound generator')
    parser.add_argument('-ar', '--ARIR_FILE', type=str, required=False,
                        help='file with FIR filter containing Array Room Impulse Responses')
    parser.add_argument('-art', '--ARIR_TYPE', type=str, required=False,
                        choices=['ARIR_SOFA', 'ARIR_MIRO', 'AS_MIRO'],
                        help='type of FIR filter file containing Array Room Impulse Responses / stream configuration')
    parser.add_argument('-arl', '--ARIR_LEVEL', type=float, required=False,
                        help='output level of renderer for Array Room Impulse Response')
    parser.add_argument('-arm', '--ARIR_MUTE', type=transform_str2bool, required=False,
                        help='output mute state of renderer for Array Room Impulse Response')
    parser.add_argument('-arr', '--ARIR_RADIAL_AMP', type=int, required=False,
                        help='maximum amplification limit in dB when generating modal radial filters')
    parser.add_argument('-hr', '--HRIR_FILE', type=str, required=False,
                        help='file with FIR filter containing Head Related Impulse Responses')
    parser.add_argument('-hrt', '--HRIR_TYPE', type=str, required=False,
                        choices=['HRIR_SOFA', 'HRIR_MIRO', 'HRIR_SSR', 'BRIR_SSR'],
                        help='type of FIR filter file containing Head Related Impulse Responses')
    parser.add_argument('-hrl', '--HRIR_LEVEL', type=float, required=False,
                        help='output level of renderer for Head Related Impulse Response')
    parser.add_argument('-hrm', '--HRIR_MUTE', type=transform_str2bool, required=False,
                        help='output mute state of renderer for Head Related Impulse Response')
    parser.add_argument('-hp', '--HPIR_FILE', type=str, required=False,
                        help='file with FIR filter containing Headphone Equalization Impulse Responses')
    # parser.add_argument('-hpt', '--HPIR_TYPE', type=str, required=False,
    #                     choices=['HPIR_FIR', 'HPIR_SOFA'],
    #                     help='type of FIR filter file containing Headphone Equalization Impulse Responses')
    parser.add_argument('-hpl', '--HPIR_LEVEL', type=float, required=False,
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
    parser.add_argument('-pfm', '--IS_PYFFTW_MODE', type=transform_str2bool, required=False,
                        help='if FFTW library should be used instead of numpy for all real-time DFT operations')
    parser.add_argument('-pfe', '--PYFFTW_EFFORT', type=str, required=False,
                        choices=['FFTW_ESTIMATE', 'FFTW_MEASURE', 'FFTW_PATIENT', 'FFTW_EXHAUSTIVE'],
                        help='effort spent during the FFTW planning stage to create the fastest possible transform')
    parser.add_argument('-ll', '--LOGGING_LEVEL', type=str, required=False,
                        choices=['NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='lowest logging level being shown and printed to the logs')
    parser.add_argument('-lp', '--LOGGING_PATH', type=str, required=False,
                        help='path of log messages being saved to')
    parser.add_argument('-SP', '--IS_SINGLE_PRECISION', type=transform_str2bool, nargs='?', const=True, required=False,
                        help='run processing with single precision (32 bit) for better performance, otherwise double '
                             'precision (64 bit)')
    # parser.add_argument('-SP', '--SINGLE_PRECISION', action=_PrecisionAction, nargs=0,
    #                     help='run processing with single precision (32 bit) for better performance')
    # parser.add_argument('-DP', '--DOUBLE_PRECISION', action=_PrecisionAction, nargs=0,
    #                     help='run processing with double precision (64 bit) for better accuracy')
    parser.add_argument('--STUDY_MODE', type=transform_str2bool, nargs='?', const=True, required=False,
                        help='run rendering mode with minimal startup time and preferential performance settings')
    parser.add_argument('--BENCHMARK_MODE', type=str, required=False,
                        choices=['PARALLEL_CLIENTS', 'PARALLEL_CONVOLVERS'],
                        help='run benchmark mode with specified method, ignoring all other parameters')
    parser.add_argument('--VALIDATION_MODE', type=str, required=False,
                        help='run validation mode against provided reference impulse response set')
    parser.add_argument('--DEVELOPER_MODE', type=transform_str2bool, nargs='?', const=True, required=False,
                        help='run development test mode')

    # parse arguments
    args = parser.parse_args()

    # update config
    from . import config
    for a in args.__dict__:
        value = args.__dict__[a]
        if value is not None:
            set_arg(config, a, value)

    print('all unnamed arguments use default values (see module `config`).')

    # manually call to update logging path
    config.LOGGING_PATH = get_absolute_from_relative_package_path(config.LOGGING_PATH)


def set_arg(ref, arg, value):
    """
    Allow for parsing of arguments and update according values in the provided reference module.

    Parameters
    ----------
    ref : module
        Python module the argument should be set with the value
    arg : str
        argument name that should be set
    value : Any
        value that should be set
    """
    try:
        value_default = getattr(ref, arg)

        # if parameter is path, transform into relative path
        if value_default is not None and not isinstance(value_default, (list, int, float)) \
                and (os.path.isfile(value_default) or os.path.isdir(value_default)):
            value_default_rel = os.path.relpath(value_default)
        else:
            value_default_rel = value_default

        if value is not None and os.path.isfile(str(value)):
            # if parameter is path, transform into absolute path
            value = os.path.abspath(value)
            value_rel = os.path.relpath(value)
        else:
            if isinstance(value_default, list):
                # if value is list
                if not ('[(' in value and ',' in value and ')]' in value):
                    print(f'{ref.__name__ + "." + arg:<30} has incorrect shape.\n'
                          f'application interrupted.', file=sys.stderr)
                    sys.exit(1)

                import ast
                value = ast.literal_eval(value)

            value_rel = value

        if value != value_default and \
                (value_default is not None or (value_default is None and str(value).upper() != 'NONE')):
            # set parameter in config
            value_rel_str = f'"{value_rel}"'
            print(f'{ref.__name__ + "." + arg:<35} using {value_rel_str:<40} (default "{value_default_rel}").')
            setattr(ref, arg, value)

    except AttributeError:
        print(f'config.{arg}   parameter unknown.\napplication interrupted.', file=sys.stderr)
        sys.exit(1)


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

    if not relative_package_path or \
            (isinstance(relative_package_path, str) and relative_package_path.strip('\'"') == ''):
        return None
    elif isinstance(relative_package_path, str):
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
        #     print(f'set process priority (OSX nice, lower values mean higher priority) from {nice} to {nice_new}.')
        # except psutil.AccessDenied:
        #     print(f'[WARNING]  process priority could not be set over {nice}.\n'
        #           ' --> Run Python with `sudo` for elevated permissions!', file=sys.stderr)
        # sleep(.2)  # to get correct output order

    lim = resource.getrlimit(resource.RLIMIT_NPROC)
    try:
        resource.setrlimit(resource.RLIMIT_NPROC, (lim[1], lim[1]))
        lim_new = resource.getrlimit(resource.RLIMIT_NPROC)
        if lim[0] == lim_new[0]:
            raise ValueError()
        print(f'set maximum number of processes the current process may create from {lim[0]} to {lim_new[0]}.')
    except ValueError:
        print(f'maximum number of processes the current process may create is {lim[0]}.')

    lim = resource.getrlimit(resource.RLIMIT_NOFILE)
    try:
        if not hasattr(config, 'PROCESS_FILE_LIMIT_MIN') or config.PROCESS_FILE_LIMIT_MIN < lim[0]:
            raise ValueError()
        resource.setrlimit(resource.RLIMIT_NOFILE, (config.PROCESS_FILE_LIMIT_MIN, config.PROCESS_FILE_LIMIT_MIN))
        lim_new = resource.getrlimit(resource.RLIMIT_NOFILE)
        if lim[0] == lim_new[0]:
            raise ValueError()
        print(f'set maximum number of open file descriptors for the current process from {lim[0]} to {lim_new[0]}.')
    except ValueError:
        print(f'maximum number of open file descriptors for the current process is {lim[0]}.')


def request_numpy_parameters():
    """
    Set `numpy` specific parameters for linked libraries like settings for automatic threading behaviour. Information
    about the utilized threading libraries is printed out afterwards.
    """

    def set_env_parameter(param, val):
        print(f'setting environment parameter {param} from {os.environ.get(param)} to {val}.')
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
    np.set_string_function(lambda ndarray:
                           f'[{["x", "C"][ndarray.flags.carray]}{["x", "F"][ndarray.flags.farray]}'
                           f'{["x", "O"][ndarray.flags.owndata]}] {ndarray.dtype} {ndarray.shape}', repr=False)


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
    implementations = {'openblas_get_num_threads':   'OpenBLAS',
                       'ATL_buildinfo':              'ATLAS',
                       'bli_thread_get_num_threads': 'BLIS',
                       'MKL_Get_Max_Threads':        'MKL', }

    for func, implementation in implementations.items():
        try:
            getattr(dll, func)
            blas.append(implementation)
        except AttributeError:
            continue

    if len(blas) > 1:
        print(f'[WARNING]  multiple BLAS/LAPACK libs loaded: {blas}')

    if len(blas) == 0:
        print(f'[WARNING]  unable to guess BLAS implementation, it is not one of: {implementations.values()}')
        print(' --> additional symbols are not loaded?!')

    link_str = 'numpy linked to'
    for impl in blas:
        if impl == 'OpenBLAS':
            dll.openblas_get_config.restype = ctypes.c_char_p
            dll.openblas_get_num_threads.restype = ctypes.c_int
            print(f'{link_str} "{impl}" (num threads: {dll.openblas_get_num_threads()})\n'
                  f' --> {dll.openblas_get_config().decode("utf8").strip()}')

        elif impl == 'BLIS':
            dll.bli_thread_get_num_threads.restype = ctypes.c_int
            print(f'{link_str} "{impl}" (num threads: {dll.bli_thread_get_num_threads()}, '
                  f'threads enabled: {dll.bli_info_get_enable_threading()}')

        elif impl == 'MKL':
            version_func = dll.mkl_get_version_string
            version_func.argtypes = (ctypes.c_char_p, ctypes.c_int)
            out_buf = ctypes.c_buffer(500)
            version_func(out_buf, 500)
            print(f'{link_str} "{impl}" (max threads: {dll.MKL_Get_Max_Threads()})\n'
                  f' --> {out_buf.value.decode("utf8").strip()}')

        elif impl == 'ATLAS':
            print(
                f'{link_str} "{impl}" (ATLAS is thread-safe, max number of threads are fixed at compile time)\n'
                f' --> {dll.ATL_buildinfo()}')

        else:
            print(f'{link_str} "{impl}"')

    if 'MKL' not in blas:
        print('[WARNING]  "MKL" version of `numpy` is not linked, which is supposed to provide best performance.',
              file=sys.stderr)


def import_fftw_wisdom(is_enforce_load=False):
    """
    Load and import gathered FFTW wisdom from provided file and set global `pyfftw` parameters according to
    configuration. If no wisdom can be imported, information is given that it will be generated before audio
    rendering starts.

    Parameters
    ----------
    is_enforce_load : bool, optional
        if loading wisdom should be enforced, so the application will be interrupted in case an error occurred
    """

    def log_error(log_str):
        print(log_str, file=sys.stderr)
        print(' --> All necessary wisdom will be generated now. This might take a while, before the rendering will '
              'start.\n --> Take care to properly terminate this application to have the gathered wisdom exported!',
              file=sys.stderr)
        sleep(.05)  # to get correct output order

    import pickle
    from . import config

    print(f'loading gathered FFTW wisdom from "{os.path.relpath(config.PYFFTW_WISDOM_FILE)}" ...')
    try:
        # load from file
        with open(config.PYFFTW_WISDOM_FILE, mode='rb') as f:
            wisdom = pickle.load(f)

        # load wisdom
        import pyfftw
        pyfftw.import_wisdom(wisdom)

        # print wisdom
        for w in wisdom:
            n = w.decode('utf-8').strip().split('\n')
            print(f' --> {len(n) - 2:>3} entries for "{n[0].strip("()")}"')

        # set global config parameters
        if not config.DEVELOPER_MODE and hasattr(config, 'PYFFTW_NUM_THREADS') \
                and pyfftw.config.NUM_THREADS != config.PYFFTW_NUM_THREADS:
            print(f'setting `pyfftw` environment parameter NUM_THREADS from {pyfftw.config.NUM_THREADS} to '
                  f'{config.PYFFTW_NUM_THREADS}.')
            pyfftw.config.NUM_THREADS = config.PYFFTW_NUM_THREADS
        else:
            print(f'`pyfftw` environment parameter NUM_THREADS is {pyfftw.config.NUM_THREADS}.')

        if not config.DEVELOPER_MODE and hasattr(config, 'PYFFTW_EFFORT') \
                and pyfftw.config.PLANNER_EFFORT != config.PYFFTW_EFFORT:
            print(f'setting `pyfftw` environment parameter {"PLANNER_EFFORT"} from {pyfftw.config.PLANNER_EFFORT} to '
                  f'{config.PYFFTW_EFFORT}.')
            pyfftw.config.PLANNER_EFFORT = config.PYFFTW_EFFORT
        else:
            print(f'`pyfftw` environment parameter PLANNER_EFFORT is {pyfftw.config.PLANNER_EFFORT}.')

    except FileNotFoundError:
        if is_enforce_load:
            print('... file not found while load was enforced.\napplication interrupted.', file=sys.stderr)
            sys.exit(1)
        log_error('... file not found.')

    except EOFError:
        if is_enforce_load:
            print('... error reading file while load was enforced.\napplication interrupted.', file=sys.stderr)
            sys.exit(1)
        log_error('... error reading file.')

        # rename existing file as backup
        backup = os.path.join(
            os.path.dirname(config.PYFFTW_WISDOM_FILE), f'BACKUP_{os.path.basename(config.PYFFTW_WISDOM_FILE)}')
        os.rename(config.PYFFTW_WISDOM_FILE, backup)
        print(f'... renamed existing file to "{os.path.relpath(backup)}".', file=sys.stderr)


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

    log_str = f'writing gathered FFTW wisdom to "{os.path.relpath(config.PYFFTW_WISDOM_FILE)}" ...'
    logger.info(log_str) if logger else print(log_str)

    with open(config.PYFFTW_WISDOM_FILE, mode='wb') as f:
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
    return f'{np.array2string(samples, precision=0, separator=", ")} samples / ' \
           f'{np.array2string(delay * 1000, precision=1, separator=", ")} ms / ' \
           f'{np.array2string(distance, precision=3, separator=", ")} m'


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

    def get_type_str():
        return f'{_type.__module__}.{_type.__name__}'

    if str_or_instance is None:
        return None
    elif isinstance(str_or_instance, str):
        if str_or_instance.upper() == 'NONE':
            return None
        try:
            # transform string into enum, will fail in case an invalid type string was given
            # noinspection PyUnresolvedReferences
            return _type[str_or_instance]
        except KeyError:
            raise ValueError(f'unknown parameter "{str_or_instance}", see `{get_type_str()}` for reference!')
    elif isinstance(str_or_instance, _type):
        return str_or_instance
    else:
        raise ValueError(f'unknown parameter type `{type(str_or_instance)}`, see `{get_type_str()}` for reference!')


def transform_str2bool(_str):
    """
    Parameters
    ----------
    _str : str or None
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
    if _str is None or _str.upper() in ('TRUE', 'YES', 'T', 'Y', '1'):
        return True
    elif _str.upper() in ('FALSE', 'NO', 'F', 'N', '0'):
        return False
    elif _str.upper() in ('TOGGLE', 'SWITCH', 'T', 'S', '-1'):
        return None
    else:
        raise ValueError(f'unknown boolean equivalent string "{_str}".')


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
    if state is None or isinstance(state, bool):
        return state

    # parse str
    if isinstance(state, str):
        # noinspection PyUnresolvedReferences
        return transform_str2bool(state.strip())

    # parse int and float
    if isinstance(state, (int, float)):
        state = int(state)
        if state == 1:
            return True
        if state == 0:
            return False
        if state == -1:
            return None

    # no match found
    log_str = f'unknown state "{state}"'
    logger.warning(log_str) if logger else print(log_str, file=sys.stderr)
    return None


def transform_into_osc_target(name):
    """
    Parameters
    ----------
    name : str
        client name that should be transformed

    Returns
    -------
    str
        simplified OSC target name
    """
    import re

    if name.startswith(__package__):  # cut package name
        name = name[len(__package__):]
    name = re.sub('\W+', '', name).lower()  # delete all non-alphanumeric characters
    return f'/{name}'


def generate_noise(shape, scale=1 / 10, dtype='float64'):
    """
    Parameters
    ----------
    shape : tuple of int
        shape of noise to generate (last axis contains normally distributed time samples)
    scale : float, optional
        numpy.random.normal scaling factor, the default value is supposed to result in amplitudes [-1, 1]
    dtype : str or numpy.dtype or type, optional
        numpy data type of generated array

    Returns
    -------
    numpy.ndarray
        generated white noise (normal distributed) with given shape

    Raises
    ------
    ValueError
        in case an unsupported data type is given

    Notes
    -----
    Generation in double precision yields better performance, since single precision is not natively supported and
    has to be generated by type casting.
    """
    import numpy as np

    if np.dtype(dtype) == np.complex128:
        return np.random.normal(loc=0, scale=scale, size=(shape[0], shape[1] * 2)).view(np.complex128)
    elif np.dtype(dtype) == np.complex64:
        return np.random.normal(loc=0, scale=scale, size=(shape[0], shape[1] * 2)).astype(np.float32).view(np.complex64)
    elif np.dtype(dtype) == np.float64:
        return np.random.normal(loc=0, scale=scale, size=shape)
    elif np.dtype(dtype) == np.float32:
        return np.random.normal(loc=0, scale=scale, size=shape).astype(np.float32)
    else:
        raise ValueError(f'unknown data type "{dtype}".')


def generate_iir_filter_fd(type_str, length_td, fs, fc, iir_order=4, is_lr=False, is_linear_phase=True,
                           is_apply_window=True):
    """
    Parameters
    ----------
    type_str : str
        filter type, see `scipy.signal.butter()` for reference (i.e. ‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’)
    length_td : int
        length of filter (number of taps in time domain)
    fs : int
        sampling frequency of filter
    fc : float
        cutoff frequency of generated highpass
    iir_order : int, optional
        equivalent IIR filter order (has to be even in case of Linkwitz-Riley filter)
    is_lr : bool, optional
        if equivalent magnitude response to Linkwitz-Riley IIR filter should be generated
    is_linear_phase : bool, optional
        if generated filter should be linear phase, otherwise will be zero phase
    is_apply_window : bool, optional
        if generated filter should have a Hanning / cosine time domain window applied

    Returns
    -------
    numpy.ndarray
        one-sided spectrum of generated FIR filter with desired properties
    """
    from scipy.signal import butter, sosfreqz
    import numpy as np
    import sound_field_analysis as sfa

    if is_lr:
        if iir_order // 2:
            raise ValueError('IIR filter order needs to be even for a Linkwitz-Riley filter.')
        # adjust order since Linkwitz-Riley filter is created from two Butterworth filters
        iir_order /= 2

    # generate IIR filter Second-Order-Sections (this is preferred over b and a coefficients due to numerical precision)
    filter_sos = butter(iir_order, fc, btype=type_str, output='sos', fs=fs)

    # calculate "equivalent" zero phase FIR filter one-sided spectrum
    _, filter_fd = sosfreqz(filter_sos, worN=np.linspace(0, fs / 2, length_td // 2 + 1), fs=fs)
    filter_fd[np.isnan(filter_fd)] = 0  # prevent NaNs

    if is_lr:
        # square to create Linkwitz-Riley type
        filter_fd *= filter_fd

    # generate Hanning / cosine window
    win_td = np.hanning(length_td)

    if is_linear_phase:
        # circular shift filter to make linear phase
        filter_fd *= sfa.gen.delay_fd(target_length_fd=filter_fd.shape[-1], delay_samples=length_td / 2)
    elif is_apply_window:
        # circular shift window to make zero phase
        win_td = np.roll(win_td, shift=int(-length_td / 2), axis=-1)

    if is_apply_window:
        # apply window to filter
        filter_fd = np.fft.rfft(np.fft.irfft(filter_fd) * win_td)

    return filter_fd


def calculate_rms(data_td, is_level=False):
    """
    Parameters
    ----------
    data_td : numpy.ndarray
        time domain data (along last axis) the root mean square value should be calculated of
    is_level : bool, optional
        if RMS value should be calculated as level in dB

    Returns
    -------
    numpy.ndarray
        root mean square values of provided time domain data
    """
    import numpy as np

    rms = np.sqrt(np.mean(np.square(data_td), axis=-1))
    if is_level:
        rms[np.nonzero(rms == 0)] = np.nan  # prevent zeros
        rms = 20 * np.log10(rms)  # transform into level
    return rms


def calculate_peak(data_td, is_level=False):
    """
    Parameters
    ----------
    data_td : numpy.ndarray
        time domain data (along last axis) the absolute peak value should be calculated of
    is_level : bool, optional
        if RMS value should be calculated as level in dB

    Returns
    -------
    numpy.ndarray
        absolute peak values of provided time domain data
    """
    import numpy as np

    peak = np.nanmax(np.abs(data_td), axis=-1)
    if is_level:
        peak[np.nonzero(peak == 0)] = np.nan  # prevent zeros
        peak = 20 * np.log10(peak)  # transform into level
    return peak


def plot_ir_and_tf(data_td_or_fd, fs, lgd_ch_ids=None, is_label_x=True, is_share_y=True, is_draw_grid=True,
                   is_etc=False, set_td_db_y=None, set_fd_db_y=None, step_db_y=5, set_fd_f_x=None,
                   is_show_blocked=None):
    """
    Parameters
    ----------
    data_td_or_fd : numpy.ndarray
        time (real) or one-sided frequency domain (complex) data that should be plotted of size [number of channels;
        number of samples or bins]
    fs : int
        sampling frequency of data
    lgd_ch_ids : array_like, optional
        aa
    is_label_x : bool, optional
        if x-axis of last plot should have a label
    is_share_y : bool, optional
        if y-axis dimensions of plots for all data channels should be shared
    is_draw_grid : bool, optional
        if grid should be drawn (time domain plot visualizes current processing block length)
    is_etc : bool, optional
        if time domain plot should be done as Energy Time Curve (y-axis in dB_FS)
    set_td_db_y : float or list of float or array_like, optional
        limit of time domain plot y-axis in dB (only in case of `is_etc`)
    set_fd_db_y : float or list of float or array_like, optional
        limit of frequency domain plot y-axis in dB
    step_db_y : float, optional
        step size of frequency (and time domain in case of `is_etc`) domain plot y-axis in dB for minor grid and
        rounding of limits
    set_fd_f_x : list of float or array_like, optional
        limit of frequency domain plot x-axis in Hz
    is_show_blocked : bool, optional
        if figure should be shown with the provided `block` status

    Returns
    -------
    matplotlib.figure.Figure
        generated plot
    """

    import numpy as np
    from matplotlib.ticker import FuncFormatter

    def _adjust_y_lim(is_fd=True):
        set_db_y = set_fd_db_y if is_fd else set_td_db_y
        lim_y = fd_lim_y if is_fd else td_lim_y
        if set_db_y is not None and len(set_db_y) == 2:
            # return provided fixed limits
            return set_db_y
        # get current data limits
        v_min, v_max = _get_y_data_lim(is_fd) if is_share_y else axes[ch, is_fd].yaxis.get_data_interval()
        # add to limits in case current data is exactly at limit
        if not v_min % step_db_y:
            v_min -= step_db_y
        if not v_max % step_db_y:
            v_max += step_db_y
        # prevent infinity
        if v_min == -np.inf or v_min == np.inf:
            v_min = -1e12
        if v_max == -np.inf or v_max == np.inf:
            v_max = 1e12
        # round limits
        v_max = step_db_y * np.ceil(v_max / step_db_y)
        if set_db_y is None:
            v_min = step_db_y * np.floor(v_min / step_db_y)
        else:
            # set minimum according to provided dynamic range under maximum
            v_min = v_max - set_db_y[0]
        # adjust according to and update global limit
        if is_share_y:
            if set_db_y is not None:
                v_min = min([lim_y[0], v_min])
            v_max = max([lim_y[1], v_max])
            lim_y[0] = v_min
            lim_y[1] = v_max
        # if is_fd:
        #     print(v_min, v_max)
        return v_min, v_max

    def _get_y_data_lim(_column):
        # get current data limits from all subplots
        v_min = min(axes[_ch, int(_column)].yaxis.get_data_interval()[0] for _ch in range(data_td.shape[0]))
        v_max = max(axes[_ch, int(_column)].yaxis.get_data_interval()[1] for _ch in range(data_td.shape[0]))
        return v_min, v_max

    # check and set provided parameters
    if is_etc and set_td_db_y is not None:
        if not isinstance(set_td_db_y, list):
            set_td_db_y = [set_td_db_y]
        assert (len(set_td_db_y) <= 2)
        if len(set_td_db_y) == 1:
            # noinspection PyTypeChecker
            assert (set_td_db_y[0] > 0)
    if set_fd_db_y is not None:
        if not isinstance(set_fd_db_y, list):
            set_fd_db_y = [set_fd_db_y]
        assert (len(set_fd_db_y) <= 2)
        if len(set_fd_db_y) == 1:
            # noinspection PyTypeChecker
            assert (set_fd_db_y[0] > 0)
    if set_fd_f_x is None:
        set_fd_f_x = [20, fs / 2]
    else:
        assert (len(set_fd_f_x) == 2)
    assert (step_db_y > 0)

    fd_lim_y = [1e12, -1e12]  # initial values
    td_lim_y = [1e12, -1e12]  # initial values
    _TD_STEM_LIM = 8  # upper limit in samples until a plt.stem() will be used instead of plt.plot()
    _FREQS_LABELED = [1, 10, 100, 1000, 10000, 100000]  # labeled frequencies
    _FREQS_LABELED.extend(set_fd_f_x)  # add labels at upper and lower frequency limit

    # check data size
    data_td_or_fd = np.atleast_2d(data_td_or_fd)
    if data_td_or_fd.ndim >= 3:
        data_td_or_fd = data_td_or_fd.squeeze()
        if data_td_or_fd.ndim >= 3:
            raise ValueError(f'plotting of data with size {data_td_or_fd.shape} not supported.')

    if lgd_ch_ids is None:
        lgd_ch_ids = range(data_td_or_fd.shape[0])
    else:
        assert (isinstance(lgd_ch_ids, (list, range, np.ndarray)))
        assert (len(lgd_ch_ids) == data_td_or_fd.shape[0])

    if np.iscomplexobj(data_td_or_fd):
        # fd data given
        data_fd = data_td_or_fd.copy()  # make copy to not alter input data
        if data_fd.shape[1] == 1:
            data_fd = np.repeat(data_fd, 2, axis=1)
        data_td = np.fft.irfft(data_fd)
    else:
        # td data given
        data_td = data_td_or_fd.copy()  # make copy to not alter input data
        if data_td.shape[1] == 1:
            data_td[:, 1] = 0
        data_fd = np.fft.rfft(data_td)
    del data_td_or_fd

    # prevent zeros
    data_fd[np.nonzero(data_fd == 0)] = np.nan
    if is_etc:
        data_td[np.nonzero(data_td == 0)] = np.nan

        # transform td data into logarithmic scale
        data_td = 20 * np.log10(np.abs(data_td))

    fig, axes = plt.subplots(nrows=data_td.shape[0], ncols=2, squeeze=False,
                             sharex='col', sharey='col' if is_share_y else False)
    for ch in range(data_td.shape[0]):
        # # # plot IR # # #
        length = len(data_td[ch])
        if length > _TD_STEM_LIM:
            axes[ch, 0].plot(np.arange(0, length), data_td[ch], linewidth=.5, color='C0')
        else:
            axes[ch, 0].stem(data_td[ch], linefmt='C0-', markerfmt='C0.', basefmt='C0-', use_line_collection=True)
        # set limits
        if is_etc:
            axes[ch, 0].set_ylim(*_adjust_y_lim(is_fd=False))  # needs to be done before setting yticks
        # set ticks and grid
        axes[ch, 0].tick_params(which='major', direction='in', top=True, bottom=True, left=True, right=True)
        axes[ch, 0].tick_params(which='minor', length=0)
        if is_draw_grid:
            from . import config
            length_2 = 2 ** np.ceil(np.log2(length))  # next power of 2
            if length > config.BLOCK_LENGTH:
                axes[ch, 0].set_xticks(np.arange(0, length + 1, length_2 // 4 if length_2 > length else length // 2),
                                       minor=False)
                axes[ch, 0].set_xticks(np.arange(0, length + 1, config.BLOCK_LENGTH), minor=True)
                axes[ch, 0].grid(True, which='both', axis='x', color='r', alpha=.4)
            else:
                axes[ch, 0].set_xticks(np.arange(0, length + 1, length_2 // 4 if length_2 > 4 else 2), minor=False)
                axes[ch, 0].grid(True, which='major', axis='x', alpha=.25)
            if is_etc:
                axes[ch, 0].grid(True, which='both', axis='y', alpha=.1)
                axes[ch, 0].set_yticks(np.arange(*axes[ch, 0].get_ylim(), step_db_y), minor=True)
        # set limits
        if length > _TD_STEM_LIM:
            axes[ch, 0].set_xlim(0, length)
        else:
            axes[ch, 0].set_xticks(np.arange(0, length, 1), minor=False)  # overwrite ticks
            axes[ch, 0].set_xlim(-.5, length - .5)
        # set labels
        if is_label_x and ch == data_td.shape[0] - 1:
            axes[ch, 0].set_xlabel('Samples')
        # set legend
        if lgd_ch_ids:
            lgd_str = f'ch {lgd_ch_ids[ch]}{" (ETC)" if is_etc else ""}'
            axes[ch, 0].legend([lgd_str], loc='upper right', fontsize='x-small')
        # set axes in background
        axes[ch, 0].set_zorder(-1)

        # # # plot spectrum # # #
        axes[ch, 1].semilogx(np.linspace(0, fs / 2, len(data_fd[ch])), 20 * np.log10(np.abs(data_fd[ch])), color='C1')
        # set limits
        axes[ch, 1].set_ylim(*_adjust_y_lim(is_fd=True))  # needs to be done before setting yticks
        # set ticks and grid
        axes[ch, 1].set_xticks(_FREQS_LABELED)
        axes[ch, 1].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x / 1000:.16g}'))
        axes[ch, 1].tick_params(which='major', direction='in', top=True, bottom=True, left=True, right=True)
        axes[ch, 1].tick_params(which='minor', length=0)
        if is_draw_grid:
            axes[ch, 1].grid(True, which='major', axis='both', alpha=.25)
            axes[ch, 1].grid(True, which='minor', axis='both', alpha=.1)
            axes[ch, 1].set_yticks(np.arange(*axes[ch, 1].get_ylim(), step_db_y), minor=True)
        # set limits
        axes[ch, 1].set_xlim(*set_fd_f_x)  # needs to be done after setting xticks
        # set labels
        if is_label_x and ch == data_td.shape[0] - 1:
            axes[ch, 1].set_xlabel('Frequency / kHz')
        # set axes in background
        axes[ch, 1].set_zorder(-1)

    # remove layout margins
    fig.tight_layout(pad=0)

    if is_show_blocked is not None:
        plt.show(block=is_show_blocked)

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
        if config.LOGGING_PATH is None:
            return
        name = os.path.join(config.LOGGING_PATH, name)
    # store all requested file types
    for ending in re.split(r'[,.;:|/\-+]+', file_type):
        file = f'{name}{os.path.extsep}{ending}'
        log_str = f'writing results to "{os.path.relpath(file)}" ...'
        logger.info(log_str) if logger else print(log_str)
        figure.savefig(file, dpi=300)
    # close figure
    plt.close(figure)


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
    with open(file_name, mode='w') as f:
        f.write(html_all)
