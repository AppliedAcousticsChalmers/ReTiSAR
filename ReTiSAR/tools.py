import logging
import os
import sys
from time import sleep

if sys.platform == "darwin":
    # prevent exception due to python not being a framework build when installed
    import matplotlib  # chosen by default not non-interactive backend 'agg' (matplotlib from conda)

    # matplotlib.use("TkAgg")  # this backend lead to complete system crashes recently on
    # matplotlib=3.1.0
    matplotlib.use("MacOSX")  # this backend seems to work fine
    del matplotlib
    import matplotlib.pyplot as plt
else:
    import matplotlib.pyplot as plt

# reset matplotlib logging level
logging.getLogger("matplotlib").setLevel(logging.INFO)


def parse_cmd_args():
    """Allow for parsing of certain command line arguments and update according values in
    `config`."""
    import argparse

    class _LicenseAction(argparse.Action):
        def __call__(self, _parser, namespace, values, option_string=None):
            print(
                open(
                    get_absolute_from_relative_package_path("LICENSE"),
                    mode="r",
                    encoding="utf-8",
                ).read()
            )
            _parser.exit()

    class _VersionAction(argparse.Action):
        def __call__(self, _parser, namespace, values, option_string=None):
            from . import __version__

            print(__version__)
            _parser.exit()

    # class _PrecisionAction(argparse.Action):
    #     def __call__(self, _parser, namespace, values, option_string=None):
    #         # False if self.dest == "DOUBLE_PRECISION"
    #         config.IS_SINGLE_PRECISION = self.dest == "SINGLE_PRECISION"

    # fmt: off
    # create parser and introduce all possible arguments
    parser = argparse.ArgumentParser(
        prog=__package__,
        description="Implementation of the Real-Time Spherical Microphone Renderer for binaural "
                    "reproduction in Python.")
    parser.add_argument("-l", "--license", action=_LicenseAction, nargs=0,
                        help="show LICENSE information and exit")
    parser.add_argument("-v", "--version", action=_VersionAction, nargs=0,
                        help="show version information and exit")
    parser.add_argument("-b", "--BLOCK_LENGTH", type=int, required=False,
                        help="block length in samples of the JACK audio server and clients in "
                             "samples")
    parser.add_argument("-irt", "--IR_TRUNCATION_LEVEL", type=float, required=False,
                        help="relative level in dB under peak to individually truncate any "
                             "impulse response set after load to save performance")
    parser.add_argument("-sh", "--SH_MAX_ORDER", type=int, required=False,
                        help="spherical harmonics order when rendering Array Room Impulse "
                             "Responses")
    parser.add_argument("-sht", "--SH_COMPENSATION_TYPE", type=str, required=False,
                        help="type of spherical harmonics processing compensation techniques, "
                             "see documentation for valid choices e.g. combinations of "
                             "{MODAL_RADIAL_FILTER,SPHERICAL_HEAD_FILTER,"
                             "SPHERICAL_HARMONICS_TAPERING,SECTORIAL_DEGREE_SELECTION,"
                             "EQUATORIAL_DEGREE_SELECTION}")
    parser.add_argument("-shp", "--SH_IS_ENFORCE_PINV", type=transform_str2bool, required=False,
                        help="if pseudo-inverse (Moore-Penrose) matrix will be used over "
                             "explicitly given sampling grid weights (only relevant for filter "
                             "sets in MIRO format)")
    parser.add_argument("-s", "--SOURCE_FILE", type=str, required=False,
                        help="file of audio being played by the application")
    parser.add_argument("-sp", "--SOURCE_POSITIONS", type=str, required=False,
                        help="source positions as list of tuple of azimuth and elevation in "
                             "degrees")
    parser.add_argument("-sap", "--SOURCE_IS_AUTO_PLAY", type=transform_str2bool, required=False,
                        help="auto play state of source audio replay")
    parser.add_argument("-sl", "--SOURCE_LEVEL", type=float, required=False,
                        help="output level in dBFS of source audio replay")
    parser.add_argument("-sm", "--SOURCE_MUTE", type=transform_str2bool, required=False,
                        help="output mute state of source audio replay")
    parser.add_argument("-gt", "--G_TYPE", type=str, required=False,
                        choices=["NOISE_WHITE", "NOISE_IIR_EM", "NOISE_IIR_EIGENMIKE",
                                 "NOISE_IIR_PINK", "NOISE_AR_PINK", "NOISE_AR_PURPLE",
                                 "NOISE_AR_BLUE", "NOISE_AR_BROWN", "NONE"],
                        help="type of algorithm used by generator to create sound")
    parser.add_argument("-gl", "--G_LEVEL", type=float, required=False,
                        help="output level in dBFS of sound generator")
    parser.add_argument("-glr", "--G_LEVEL_REL", type=transform_str2floats, required=False,
                        help="output level in dBFS relative between ports of sound generator")
    parser.add_argument("-gm", "--G_MUTE", type=transform_str2bool, required=False,
                        help="output mute state of sound generator")
    parser.add_argument("-ar", "--ARIR_FILE", type=str, required=False,
                        help="file with FIR filter containing Array Room Impulse Responses")
    parser.add_argument("-art", "--ARIR_TYPE", type=str, required=False,
                        choices=["ARIR_SOFA", "ARIR_MIRO", "AS_MIRO"],
                        help="type of FIR filter file containing Array Room Impulse Responses / "
                             "stream configuration")
    parser.add_argument("-arl", "--ARIR_LEVEL", type=float, required=False,
                        help="output level in dBFS of renderer for Array Room Impulse Response")
    parser.add_argument("-arm", "--ARIR_MUTE", type=transform_str2bool, required=False,
                        help="output mute state of renderer for Array Room Impulse Response")
    parser.add_argument("-arr", "--ARIR_RADIAL_AMP", type=int, required=False,
                        help="maximum amplification limit in dB when generating modal radial "
                             "filters")
    parser.add_argument("-hr", "--HRIR_FILE", type=str, required=False,
                        help="file with FIR filter containing Head Related Impulse Responses")
    parser.add_argument("-hrt", "--HRIR_TYPE", type=str, required=False,
                        choices=["HRIR_SOFA", "HRIR_MIRO", "HRIR_SSR", "BRIR_SSR"],
                        help="type of FIR filter file containing Head Related Impulse Responses")
    parser.add_argument("-hrl", "--HRIR_LEVEL", type=float, required=False,
                        help="output level in dBFS of renderer for Head Related Impulse Response")
    parser.add_argument("-hrm", "--HRIR_MUTE", type=transform_str2bool, required=False,
                        help="output mute state of renderer for Head Related Impulse Response")
    parser.add_argument("-hrd", "--HRIR_DELAY", type=float, required=False,
                        help="input buffer delay in ms of renderer for Head Related Impulse "
                             "Responses")
    parser.add_argument("-hp", "--HPCF_FILE", type=str, required=False,
                        help="file with FIR filter containing Headphone Compensation Filter")
    # parser.add_argument("-hpt", "--HPCF_TYPE", type=str, required=False,
    #                     choices=["HPCF_FIR"],
    #                     help="type of FIR filter file containing Headphone Compensation Filter")
    parser.add_argument("-hpl", "--HPCF_LEVEL", type=float, required=False,
                        help="output level in dBFS of renderer for Headphone Equalization Impulse "
                             "Response")
    parser.add_argument("-hpm", "--HPCF_MUTE", type=transform_str2bool, required=False,
                        help="output mute state of renderer for Headphone Equalization Impulse "
                             "Response")
    parser.add_argument("-t", "--TRACKER_PORT", type=str, required=False,
                        help="system specific path to tracker port to read data from")
    parser.add_argument("-tt", "--TRACKER_TYPE", type=str, required=False,
                        choices=["NONE", "AUTO_ROTATE", "POLHEMUS_PATRIOT", "POLHEMUS_FASTRACK",
                                 "RAZOR_AHRS"],
                        help="type information of hardware providing head tracking data")
    parser.add_argument("-r", "--REMOTE_OSC_PORT", type=int, required=False,
                        help="port to receive Open Sound Control remote messages")
    parser.add_argument("-pfm", "--IS_PYFFTW_MODE", type=transform_str2bool, required=False,
                        help="if FFTW library should be used instead of numpy for all real-time "
                             "DFT operations")
    parser.add_argument("-pfe", "--PYFFTW_EFFORT", type=str, required=False,
                        choices=["FFTW_ESTIMATE", "FFTW_MEASURE", "FFTW_PATIENT",
                                 "FFTW_EXHAUSTIVE"],
                        help="effort spent during the FFTW planning stage to create the fastest "
                             "possible transform")
    parser.add_argument("-pfl", "--PYFFTW_LEGACY_FILE", type=str, required=False,
                        help="file with FFTW wisdom that will be accepted without valid signature")
    parser.add_argument("-ll", "--LOGGING_LEVEL", type=str, required=False,
                        choices=["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR"],
                        help="lowest logging level being shown and printed to the logs")
    parser.add_argument("-lp", "--LOGGING_PATH", type=str, required=False,
                        help="path of log messages being saved to")
    parser.add_argument("-SP", "--IS_SINGLE_PRECISION", type=transform_str2bool, nargs="?",
                        const=True, required=False,
                        help="run processing with single precision (32 bit) for better "
                             "performance, otherwise double precision (64 bit)")
    # parser.add_argument("-SP", "--SINGLE_PRECISION", action=_PrecisionAction, nargs=0,
    #                     help="run processing with single precision (32 bit) for better "
    #                          "performance")
    # parser.add_argument("-DP", "--DOUBLE_PRECISION", action=_PrecisionAction, nargs=0,
    #                     help="run processing with double precision (64 bit) for better accuracy")
    parser.add_argument("--STUDY_MODE", type=transform_str2bool, nargs="?", const=True,
                        required=False,
                        help="run rendering mode with minimal startup time and preferential "
                             "performance settings")
    parser.add_argument("--BENCHMARK_MODE", type=str, required=False,
                        choices=["PARALLEL_CLIENTS", "PARALLEL_CONVOLVERS"],
                        help="run benchmark mode with specified method, ignoring all other "
                             "parameters")
    parser.add_argument("--VALIDATION_MODE", type=str, required=False,
                        help="run validation mode against provided reference impulse response set")
    parser.add_argument("--DEVELOPER_MODE", type=transform_str2bool, nargs="?", const=True,
                        required=False,
                        help="run development test mode")
    # fmt: on

    # parse arguments
    args = parser.parse_args()

    # update config
    from . import config

    for a in args.__dict__:
        value = args.__dict__[a]
        if value is not None:
            set_arg(config, a, value)

    print("all unnamed arguments use default values (see module `config`).")

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
        if (
            value_default is not None
            and not isinstance(value_default, (list, int, float))
            and (os.path.isfile(value_default) or os.path.isdir(value_default))
        ):
            value_default_rel = os.path.relpath(value_default)
        else:
            value_default_rel = value_default

        if value is not None and os.path.isfile(str(value)):
            # if parameter is path, transform into absolute path
            value = os.path.abspath(value)
            value_rel = os.path.relpath(value)
        else:
            if isinstance(value_default, list) and isinstance(value, str):
                # if parameter is list and value is string
                if not ("[(" in value and "," in value and ")]" in value):
                    print(
                        f'{ref.__name__ + "." + arg:<30} has incorrect shape.\n'
                        f"application interrupted.",
                        file=sys.stderr,
                    )
                    sys.exit(1)

                import ast

                value = ast.literal_eval(value)

            value_rel = value

        if value != value_default and (
            value_default is not None
            or (value_default is None and str(value).upper() != "NONE")
        ):
            # set parameter in config
            value_rel_str = f'"{value_rel}"'
            print(
                f'{ref.__name__ + "." + arg:<35} using {value_rel_str:<40} (default '
                f'"{value_default_rel}").'
            )
            setattr(ref, arg, value)

    except AttributeError:
        print(
            f"config.{arg}   parameter unknown.\napplication interrupted.",
            file=sys.stderr,
        )
        sys.exit(1)


def get_is_debug():
    """
    The current implementation works fine in PyCharm, but might not work from command line or
    other IDEs.

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


def get_system_name():
    """
    Returns
    -------
    str
        system node name reported by `platform`
    """
    import platform

    return platform.node()


def get_absolute_from_relative_package_path(relative_package_path):
    """
    Parameters
    ----------
    relative_package_path : str, list of str or None
        path to a resource relative to the package base directory

    Returns
    -------
    str, list of str or None
        absolute system path to resource
    """
    if relative_package_path is None:
        return

    _PACKAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
    if not relative_package_path or (
        isinstance(relative_package_path, str)
        and relative_package_path.strip("'\"") == ""
    ):
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
    Set session specific limitations for the current process like number of processes and open
    files to its maximum possible value. If the individual limit could not be set only its
    current value is printed.
    """
    import resource
    from . import config

    print("trying to adjust process parameters ...")

    # noinspection PyUnresolvedReferences
    if hasattr(config, "PROCESS_PRIORITY") and config.PROCESS_PRIORITY != 0:
        print(
            "[WARNING]  setting of process priority currently not implemented.",
            file=sys.stderr,
        )
        # import psutil
        #
        # sleep(0.2)  # to get correct output order
        # p = psutil.Process(os.getpid())
        # nice = p.nice()
        # try:
        #     p.nice(-config.PROCESS_PRIORITY)  # negative value of NICE on OSX !!!
        #     nice_new = p.nice()
        #     if nice == nice_new:
        #         raise PermissionError()
        #     print(
        #         f"set process priority (OSX nice, lower values mean higher priority) from "
        #         f"{nice} to {nice_new}."
        #     )
        # except psutil.AccessDenied:
        #     print(
        #         f"[WARNING]  process priority could not be set over {nice}.\n"
        #         " --> Run Python with `sudo` for elevated permissions!",
        #         file=sys.stderr,
        #     )
        # sleep(0.2)  # to get correct output order

    lim = resource.getrlimit(resource.RLIMIT_NPROC)
    try:
        resource.setrlimit(resource.RLIMIT_NPROC, (lim[1], lim[1]))
        lim_new = resource.getrlimit(resource.RLIMIT_NPROC)
        if lim[0] == lim_new[0]:
            raise ValueError()
        print(
            f"set maximum number of processes the current process may create from {lim[0]} to "
            f"{lim_new[0]}."
        )
    except ValueError:
        print(
            f"maximum number of processes the current process may create is {lim[0]}."
        )

    lim = resource.getrlimit(resource.RLIMIT_NOFILE)
    try:
        if (
            not hasattr(config, "PROCESS_FILE_LIMIT_MIN")
            or config.PROCESS_FILE_LIMIT_MIN < lim[0]
        ):
            raise ValueError()
        resource.setrlimit(
            resource.RLIMIT_NOFILE,
            (config.PROCESS_FILE_LIMIT_MIN, config.PROCESS_FILE_LIMIT_MIN),
        )
        lim_new = resource.getrlimit(resource.RLIMIT_NOFILE)
        if lim[0] == lim_new[0]:
            raise ValueError()
        print(
            f"set maximum number of open file descriptors for the current process from {lim[0]} to "
            f"{lim_new[0]}."
        )
    except ValueError:
        print(
            f"maximum number of open file descriptors for the current process is {lim[0]}."
        )


# noinspection PyTypeChecker
def request_numpy_parameters():
    """
    Set `numpy` specific parameters for linked libraries like settings for automatic threading
    behaviour. Information about the utilized threading libraries is printed out afterwards.
    """

    def set_env_parameter(param, val):
        print(
            f"setting environment parameter {param} from {os.environ.get(param)} to {val}."
        )
        os.environ[param] = val

    from . import config

    print("trying to adjust numpy parameters ...")

    # these variables need to be set before `numpy` is imported the first time
    set_env_parameter("OMP_DYNAMIC", config.NUMPY_OMP_DYNAMIC.__str__())
    set_env_parameter("OMP_NUM_THREADS", config.NUMPY_OMP_NUM_THREADS.__str__())
    set_env_parameter("MKL_DYNAMIC", config.NUMPY_MKL_DYNAMIC.__str__())
    set_env_parameter("MKL_NUM_THREADS", config.NUMPY_MKL_NUM_THREADS.__str__())

    # set_env_parameter('OMP_NESTED', 'TRUE')  # no positive effect on performance
    # set_env_parameter('NUMEXPR_NUM_THREADS', '1')  # no positive effect on performance
    # set_env_parameter('OPENBLAS_NUM_THREADS', '1')  # no positive effect on performance
    # set_env_parameter('VECLIB_MAXIMUM_THREADS', '1')  # no positive effect on performance
    # set_env_parameter('MKL_DOMAIN_NUM_THREADS', '"MKL_FFT=1"')  # no positive effect on perf.

    print_numpy_info()

    import numpy as np

    # adjust numpy settings to throw exceptions for all potential error instances (division by
    # zero, overflow, underflow and invalid operation (typically due to a NaN)
    # this requires that certain operations where exceptions are known need to be called while
    # temporarily ignoring the specific warning:
    # with np.errstate(invalid="ignore"):
    np.seterr(all="raise")

    # show shape when printing np.ndarray (useful while debugging)
    np.set_string_function(
        f=lambda ndarray: f'[{["x", "C"][ndarray.flags.carray]}{["x", "F"][ndarray.flags.farray]}'
        f'{["x", "O"][ndarray.flags.owndata]}] {ndarray.dtype} {ndarray.shape}',
        repr=False,
    )


def print_numpy_info():
    """
    Prints out numpy information about how `numpy` is linked by checking what symbols are defined
    when loading the numpy modules using BLAS.

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
    implementations = {
        "openblas_get_num_threads": "OpenBLAS",
        "ATL_buildinfo": "ATLAS",
        "bli_thread_get_num_threads": "BLIS",
        "MKL_Get_Max_Threads": "MKL",
    }

    for func, implementation in implementations.items():
        try:
            getattr(dll, func)
            blas.append(implementation)
        except AttributeError:
            continue

    if len(blas) > 1:
        print(f"[WARNING]  multiple BLAS/LAPACK libs loaded: {blas}")

    if len(blas) == 0:
        print(
            f"[WARNING]  unable to guess BLAS implementation, it is not one of: "
            f"{implementations.values()}"
        )
        print(" --> additional symbols are not loaded?!")

    link_str = "numpy linked to"
    for impl in blas:
        if impl == "OpenBLAS":
            dll.openblas_get_config.restype = ctypes.c_char_p
            dll.openblas_get_num_threads.restype = ctypes.c_int
            print(
                f'{link_str} "{impl}" (num threads: {dll.openblas_get_num_threads()})\n'
                f' --> {dll.openblas_get_config().decode("utf8").strip()}'
            )

        elif impl == "BLIS":
            dll.bli_thread_get_num_threads.restype = ctypes.c_int
            print(
                f'{link_str} "{impl}" (num threads: {dll.bli_thread_get_num_threads()}, '
                f"threads enabled: {dll.bli_info_get_enable_threading()}"
            )

        elif impl == "MKL":
            version_func = dll.mkl_get_version_string
            version_func.argtypes = (ctypes.c_char_p, ctypes.c_int)
            out_buf = ctypes.c_buffer(500)
            version_func(out_buf, 500)
            print(
                f'{link_str} "{impl}" (max threads: {dll.MKL_Get_Max_Threads()})\n'
                f' --> {out_buf.value.decode("utf8").strip()}'
            )

        elif impl == "ATLAS":
            print(
                f'{link_str} "{impl}" (ATLAS is thread-safe, max number of threads are fixed at '
                f"compile time)\n --> {dll.ATL_buildinfo()}"
            )

        else:
            print(f'{link_str} "{impl}"')

    if "MKL" not in blas:
        print(
            '[WARNING]  "MKL" version of `numpy` is not linked, which is supposed to provide best '
            "performance.",
            file=sys.stderr,
        )


# noinspection PyUnresolvedReferences
def import_fftw_wisdom(is_enforce_load=False):
    """
    Load and import gathered FFTW wisdom from provided file and set global `pyfftw` parameters
    according to configuration. If no wisdom can be imported, information is given that it will
    be generated before audio rendering starts.

    Parameters
    ----------
    is_enforce_load : bool, optional
        if loading wisdom should be enforced, so the application will be interrupted in case an
        error occurred
    """

    def log_error(log_str, is_enforce_exit=False):
        if is_enforce_load or is_enforce_exit:
            print(
                f"... {log_str}{' while load was enforced' if is_enforce_load else ''}."
                f"\napplication interrupted.",
                file=sys.stderr,
            )
            sys.exit(1)
        else:
            print(f"... {log_str}.", file=sys.stderr)
            print(
                " --> All necessary wisdom will be generated now. This might take a while, before "
                "the rendering will start.\n --> Take care to properly terminate this application "
                "to have the gathered wisdom exported!",
                file=sys.stderr,
            )
            sleep(0.05)  # to get correct output order

    from . import config
    from hashlib import blake2b
    import hmac
    import pickle
    import pyfftw

    is_legacy_load = config.PYFFTW_LEGACY_FILE is not None
    if is_legacy_load:
        file_name = config.PYFFTW_LEGACY_FILE
    else:
        file_name = config.PYFFTW_WISDOM_FILE

    print(f'loading gathered FFTW wisdom from "{os.path.relpath(file_name)}" ...')
    try:
        with open(file_name, mode="rb") as file:
            # read file
            wisdom = file.read().rsplit(b"\n\n")

            if is_legacy_load:
                if len(wisdom) > 1:
                    print(
                        f"... found FFTW wisdom file and ignoring the signature.",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"... found legacy FFTW wisdom file without signature.",
                        file=sys.stderr,
                    )
                sleep(0.05)  # to get correct output order
                wisdom = wisdom[0]
            else:
                if len(wisdom) > 1:
                    # extract signature
                    digest, wisdom = wisdom

                    # strip newlines, just to be sure
                    wisdom = wisdom.strip(b"\n")

                    # compute expected hash
                    expected_digest = hmac.new(
                        key=get_system_name().encode(), msg=wisdom, digestmod=blake2b
                    ).digest()

                    # verify hash with signature
                    if hmac.compare_digest(digest, expected_digest):
                        print(f"... found valid FFTW wisdom file signature.")
                    else:
                        log_error(
                            "found invalid FFTW wisdom file signature",
                            is_enforce_exit=True,
                        )
                else:
                    # no signature given
                    log_error(
                        "found no FFTW wisdom file signature", is_enforce_exit=True
                    )

            # load wisdom
            # noinspection PickleLoad
            wisdom = pickle.loads(wisdom)  # warning prevented with verification

        # import and print wisdom
        pyfftw.import_wisdom(wisdom)
        for w in wisdom:
            n = w.decode("utf-8").strip().split("\n")
            print(f' --> {len(n) - 2:>3} entries for "{n[0].strip("()")}"')

    except FileNotFoundError:
        log_error("file not found")

    except ValueError:
        log_error("file in unsupported pickle protocol")

    except EOFError:
        log_error("error reading file")

    # set global config parameters
    if (
        not config.DEVELOPER_MODE
        and hasattr(config, "PYFFTW_NUM_THREADS")
        and pyfftw.config.NUM_THREADS != config.PYFFTW_NUM_THREADS
    ):
        print(
            f"setting `pyfftw` environment parameter NUM_THREADS from "
            f"{pyfftw.config.NUM_THREADS} to {config.PYFFTW_NUM_THREADS}."
        )
        pyfftw.config.NUM_THREADS = config.PYFFTW_NUM_THREADS
    else:
        print(
            f"`pyfftw` environment parameter NUM_THREADS is {pyfftw.config.NUM_THREADS}."
        )

    if (
        not config.DEVELOPER_MODE
        and hasattr(config, "PYFFTW_EFFORT")
        and pyfftw.config.PLANNER_EFFORT != config.PYFFTW_EFFORT
    ):
        print(
            f'setting `pyfftw` environment parameter {"PLANNER_EFFORT"} from '
            f"{pyfftw.config.PLANNER_EFFORT} to {config.PYFFTW_EFFORT}."
        )
        pyfftw.config.PLANNER_EFFORT = config.PYFFTW_EFFORT
    else:
        print(
            f"`pyfftw` environment parameter PLANNER_EFFORT is {pyfftw.config.PLANNER_EFFORT}."
        )


def export_fftw_wisdom(logger):
    """
    Write gathered FFTW wisdom to provided file for later import.

    This file will be signed in order to ensure integrity according to
    https://pycharm-security.readthedocs.io/en/latest/checks/PIC100.html.

    Parameters
    ----------
    logger : logging.Logger or None
        instance to provide identical logging behaviour as the calling process
    """
    from . import config
    from hashlib import blake2b
    import hmac
    import pickle
    import pyfftw

    log_str = f'writing gathered FFTW wisdom to "{os.path.relpath(config.PYFFTW_WISDOM_FILE)}" ...'
    logger.info(log_str) if logger else print(log_str)

    # rename existing file as backup
    if os.path.isfile(config.PYFFTW_WISDOM_FILE):
        backup = os.path.join(
            os.path.dirname(config.PYFFTW_WISDOM_FILE),
            f"BACKUP_{os.path.basename(config.PYFFTW_WISDOM_FILE)}",
        )
        os.replace(src=config.PYFFTW_WISDOM_FILE, dst=backup)
        log_str = f'... renamed existing file to "{os.path.relpath(backup)}".'
        logger.info(log_str) if logger else print(log_str)

    # generate byte wisdom data
    # enforcing pickle protocol version 4 for compatibility between Python 3.7 and 3.8
    wisdom = pickle.dumps(obj=pyfftw.export_wisdom(), protocol=4)

    # generate byte wisdom hash
    digest = hmac.new(
        key=get_system_name().encode(), msg=wisdom, digestmod=blake2b
    ).digest()

    # write data to file with hash as header
    with open(config.PYFFTW_WISDOM_FILE, mode="wb") as file:
        file.write(digest + b"\n\n" + wisdom)


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
        generated string with printed values in absolute samples as well as according delay in
        milliseconds and traveled sound distance in meters
    """
    import numpy as np

    # calculate according time delay and distances
    samples = np.asarray(np.abs(samples), dtype=np.int_)
    delay = samples / fs  # in seconds
    distance = delay * SPEED_OF_SOUND  # in meter

    # generate string
    return (
        f'{np.array2string(samples, separator=", ")} samples / '
        f'{np.array2string(delay * 1000, precision=1, separator=", ", floatmode="fixed")} ms / '
        f'{np.array2string(distance, precision=3, separator=", ", floatmode="fixed")} m'
    )


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
        return f"{_type.__module__}.{_type.__name__}"

    if str_or_instance is None:
        return None
    elif isinstance(str_or_instance, str):
        if str_or_instance.upper() == "NONE":
            return None
        try:
            # transform string into enum, will fail in case an invalid type string was given
            # noinspection PyUnresolvedReferences
            return _type[str_or_instance]
        except KeyError:
            raise ValueError(
                f'unknown parameter "{str_or_instance}", see `{get_type_str()}` for reference!'
            )
    elif isinstance(str_or_instance, _type):
        return str_or_instance
    else:
        raise ValueError(
            f"unknown parameter type `{type(str_or_instance)}`, see `{get_type_str()}` for "
            f"reference!"
        )


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
    if _str is None or _str.upper() in ("TRUE", "YES", "T", "Y", "1"):
        return True
    elif _str.upper() in ("FALSE", "NO", "F", "N", "0"):
        return False
    elif _str.upper() in ("TOGGLE", "SWITCH", "T", "S", "-1"):
        return None
    else:
        raise ValueError(f'unknown boolean equivalent string "{_str}".')


def transform_str2floats(_str):
    """
    Parameters
    ----------
    _str : str
        equivalent string to be transformed into a list of floats

    Returns
    -------
    list of float
        list of floats transformed from equivalent string
    """
    # replace potential valid inputs
    _str = (
        _str.replace(",", " ")
        .replace(";", " ")
        .replace('"', " ")
        .replace("[", " ")
        .replace("]", " ")
        .replace("{", " ")
        .replace("}", " ")
    )

    # # parse to ndarray
    # import numpy as np
    # return np.fromstring(_str, dtype=np.float32, sep=' ')

    # parse to list of stings
    _list = [s for s in _str.split(" ") if s]  # delete empty strings
    # parse to list of floats
    return [float(f) for f in _list]


def transform_into_state(state, logger=None):
    """
    Parameters
    ----------
    state : bool, int, float, str or None
        state value in compatible format for which a mapping will be achieved, if an invalid value
        is given a warning will be logged and `None` returned
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
        name = name[len(__package__) :]
    name = re.sub(r"\W+", "", name).lower()  # delete all non-alphanumeric characters
    return f"/{name}"


def transform_into_wrapped_angles(azim, elev, tilt, is_deg=True, deg_round_precision=0):
    """
    Parameters
    ----------
    azim : float
        azimuth angle (will be wrapped to -180..180 degrees)
    elev : float
        elevation angle (will be wrapped to -90..90 degrees)
    tilt : float
        tilt angle (will be wrapped to -180..180 degrees)
    is_deg : bool, optional
        if provided and delivered values are in degrees, radians otherwise
    deg_round_precision : int, optional
        number of decimals to round to (only in case of angles in degrees)

    Returns
    -------
    list of float
        azimuth, elevation and tilt angles in degrees or radians being wrapped
    """
    if is_deg:
        _AZIM_WRAP = 360
        _ELEV_WRAP = 180
        _TILT_WRAP = 360
    else:
        import numpy as np

        _AZIM_WRAP = 2 * np.pi
        _ELEV_WRAP = np.pi
        _TILT_WRAP = 2 * np.pi

    azim = azim % _AZIM_WRAP
    elev = elev % _ELEV_WRAP
    tilt = tilt % _TILT_WRAP

    if azim > _AZIM_WRAP / 2:
        azim -= _AZIM_WRAP
    if elev > _ELEV_WRAP / 2:
        elev -= _ELEV_WRAP
    if tilt > _TILT_WRAP / 2:
        tilt -= _TILT_WRAP

    if is_deg:
        azim = round(azim, ndigits=deg_round_precision)
        elev = round(elev, ndigits=deg_round_precision)
        tilt = round(tilt, ndigits=deg_round_precision)

    return [azim, elev, tilt]


def _set_noise_generator():
    """
    Generate a `SFC64` number generator instance since it yields the best performance, see
    https://numpy.org/doc/1.18/reference/random/performance.html.

    Returns
    -------
    numpy.random.Generator
        Random number generator instance to be reused during rendering for best real-time
        performance
    """

    from numpy.random import Generator, SFC64

    return Generator(SFC64())


def generate_noise(shape, scale=1 / 10, dtype="float64"):
    """
    Parameters
    ----------
    shape : tuple of int
        shape of noise to generate (last axis contains normally distributed time samples)
    scale : float, optional
        numpy.random.normal scaling factor, the default value is supposed to result in amplitudes
        [-1, 1]
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
    """
    import numpy as np

    if np.dtype(dtype) in [np.float32, np.float64]:
        return scale * _RNG.standard_normal(size=shape, dtype=dtype)

    elif np.dtype(dtype) in [np.complex64, np.complex128]:
        return (
            scale
            * _RNG.standard_normal(
                size=(shape[0], shape[1] * 2),
                dtype=np.float32 if np.dtype(dtype) == np.complex64 else np.float64,
            ).view(dtype)
        )

    else:
        raise ValueError(f'unknown data type "{dtype}".')


def generate_iir_filter_fd(
    type_str,
    length_td,
    fs,
    fc,
    iir_order=4,
    is_lr=False,
    is_linear_phase=True,
    is_apply_window=True,
):
    """
    Parameters
    ----------
    type_str : str
        filter type, see `scipy.signal.butter()` for reference, e.g. ‘lowpass’, ‘highpass’,
        ‘bandpass’, ‘bandstop’
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
            raise ValueError(
                "IIR filter order needs to be even for a Linkwitz-Riley filter."
            )
        # adjust order since Linkwitz-Riley filter is created from two Butterworth filters
        iir_order /= 2

    # generate IIR filter Second-Order-Sections (this is preferred over b and a coefficients due
    # to numerical precision)
    filter_sos = butter(iir_order, fc, btype=type_str, output="sos", fs=fs)

    # calculate "equivalent" zero phase FIR filter one-sided spectrum
    _, filter_fd = sosfreqz(
        filter_sos, worN=np.linspace(0, fs / 2, length_td // 2 + 1), fs=fs
    )
    filter_fd[np.isnan(filter_fd)] = 0  # prevent NaNs

    if is_lr:
        # square to create Linkwitz-Riley type
        filter_fd *= filter_fd

    # generate Hanning / cosine window
    win_td = np.hanning(length_td)

    if is_linear_phase:
        # circular shift filter to make linear phase
        filter_fd *= sfa.gen.delay_fd(
            target_length_fd=filter_fd.shape[-1], delay_samples=length_td / 2
        )
    elif is_apply_window:
        # circular shift window to make zero phase
        win_td = np.roll(win_td, shift=int(-length_td / 2), axis=-1)

    if is_apply_window:
        # apply window to filter
        filter_fd = np.fft.rfft(np.fft.irfft(filter_fd) * win_td)

    return filter_fd


def calculate_rms(data, is_level=False):
    """
    Parameters
    ----------
    data : numpy.ndarray
        time or frequency domain data (along last axis)
    is_level : bool, optional
        if RMS value should be calculated as level in dB

    Returns
    -------
    numpy.ndarray
        root mean square values of provided data
    """
    import numpy as np

    if np.iscomplexobj(data):
        rms = np.sqrt(
            np.sum(np.square(np.abs(data)), axis=-1) / np.square(data.shape[-1])
        )
    else:
        rms = np.sqrt(np.mean(np.square(np.abs(data)), axis=-1))
    if is_level:
        rms[np.nonzero(rms == 0)] = np.nan  # prevent zeros
        rms = 20 * np.log10(rms)  # transform into level
        # rms[np.isnan(rms)] = np.NINF  # transform zeros into -infinity
        rms[np.isnan(rms)] = -200  # transform zeros into -200 dB
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
        # peak[np.isnan(peak)] = np.NINF  # transform zeros into -infinity
        peak[np.isnan(peak)] = -200  # transform zeros into -200 dB
    return peak


def plot_ir_and_tf(
    data_td_or_fd,
    fs,
    lgd_ch_ids=None,
    is_label_x=True,
    is_share_y=True,
    is_draw_grid=True,
    is_etc=False,
    set_td_db_y=None,
    set_fd_db_y=None,
    step_db_y=5,
    set_fd_f_x=None,
    is_draw_td=True,
    is_draw_fd=True,
    is_show_blocked=None,
):
    """
    Parameters
    ----------
    data_td_or_fd : numpy.ndarray
        time (real) or one-sided frequency domain (complex) data that should be plotted of size
        [number of channels; number of samples or bins]
    fs : int
        sampling frequency of data
    lgd_ch_ids : array_like, optional
        IDs that should be printed in the legend as individual channel names of size
        [number of channels] (range from 0 if nothing is specified)
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
        step size of frequency (and time domain in case of `is_etc`) domain plot y-axis in dB for
        minor grid and rounding of limits
    set_fd_f_x : list of float or array_like, optional
        limit of frequency domain plot x-axis in Hz
    is_draw_td : bool, optional
        if figure should contain time domain plot
    is_draw_fd : bool, optional
        if figure should contain frequency domain plot
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
        col = fd_col if is_fd else td_col
        set_db_y = set_fd_db_y if is_fd else set_td_db_y
        lim_y = fd_lim_y if is_fd else td_lim_y
        if set_db_y is not None and len(set_db_y) == 2:
            # return provided fixed limits
            return set_db_y
        # get current data limits
        v_min, v_max = (
            _get_y_data_lim(col)
            if is_share_y
            else axes[ch, int(col)].yaxis.get_data_interval()
        )
        # add to limits in case current data is exactly at limit
        if not v_min % step_db_y:
            v_min -= step_db_y
        if not v_max % step_db_y:
            v_max += step_db_y
        # prevent infinity
        if v_min == np.NINF or v_min == np.Inf:
            v_min = -1e12
        if v_max == np.NINF or v_max == np.Inf:
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
        v_min = min(
            axes[_ch, int(_column)].yaxis.get_data_interval()[0]
            for _ch in range(data_td.shape[0])
        )
        v_max = max(
            axes[_ch, int(_column)].yaxis.get_data_interval()[1]
            for _ch in range(data_td.shape[0])
        )
        return v_min, v_max

    def _check_y_db_param(_db_y):
        # check provided y-axis (time or frequency domain) limits
        if not isinstance(_db_y, list):
            _db_y = [_db_y]
        if len(_db_y) > 2:
            raise ValueError(
                f"number of Decibel axis limits ({len(_db_y)}) is greater 2."
            )
        if len(_db_y) == 1 and _db_y[0] <= 0:
            raise ValueError(
                f"value of single Decibel axis limit ({_db_y[0]}) is smaller equal 0."
            )
        return _db_y

    # check and set provided parameters
    if is_etc and set_td_db_y is not None:
        set_td_db_y = _check_y_db_param(set_td_db_y)
    if set_fd_db_y is not None:
        set_fd_db_y = _check_y_db_param(set_fd_db_y)
    if set_fd_f_x is None:
        set_fd_f_x = [20, fs / 2]
    elif len(set_fd_f_x) != 2:
        raise ValueError(
            f"number of frequency axis limits ({len(set_fd_f_x)}) is not 2."
        )
    if step_db_y <= 0:
        raise ValueError(f"step size of Decibel axis ({step_db_y}) is smaller equal 0.")

    fd_lim_y = [1e12, -1e12]  # initial values
    td_lim_y = [1e12, -1e12]  # initial values
    _TD_STEM_LIM = 8  # upper limit in samples until a plt.stem() will be used instead of plt.plot()
    _FREQS_LABELED = [1, 10, 100, 1000, 10000, 100000]  # labeled frequencies
    _FREQS_LABELED.extend(set_fd_f_x)  # add labels at upper and lower frequency limit

    td_col = 0
    fd_col = 1 if is_draw_td else 0

    # check provided data size
    data_td_or_fd = np.atleast_2d(data_td_or_fd)
    if data_td_or_fd.ndim >= 3:
        data_td_or_fd = data_td_or_fd.squeeze()
        if data_td_or_fd.ndim >= 3:
            raise ValueError(
                f"plotting of data with size {data_td_or_fd.shape} is not supported."
            )

    # check provided legend IDs
    if lgd_ch_ids is None:
        if data_td_or_fd.shape[0] > 1:
            lgd_ch_ids = range(data_td_or_fd.shape[0])
    elif not isinstance(lgd_ch_ids, (list, range, np.ndarray)):
        raise TypeError(
            f"legend channel IDs of type {type(lgd_ch_ids)} are not supported."
        )
    elif len(lgd_ch_ids) != data_td_or_fd.shape[0]:
        raise ValueError(
            f"length of legend channel IDs ({len(lgd_ch_ids)}) does not match "
            f"the size of the data ({data_td_or_fd.shape[0]})."
        )

    if np.iscomplexobj(data_td_or_fd):
        # fd data given
        data_fd = data_td_or_fd.copy()  # make copy to not alter input data
        if data_td_or_fd.shape[1] == 1:
            data_fd = np.repeat(data_fd, 2, axis=1)
        data_td = np.fft.irfft(data_fd, axis=1)
        if data_td_or_fd.shape[1] == 1:
            data_td = data_td[:, :1]
    else:
        # td data given
        data_td = data_td_or_fd.copy()  # make copy to not alter input data
        # if data_td.shape[1] == 1:
        #     data_td[:, 1] = 0
        data_fd = np.fft.rfft(data_td, axis=1)
    del data_td_or_fd

    # prevent zeros
    data_fd[np.nonzero(data_fd == 0)] = np.nan
    if is_etc:
        data_td[np.nonzero(data_td == 0)] = np.nan
        # transform td data into logarithmic scale
        data_td = 20 * np.log10(np.abs(data_td))

    fig, axes = plt.subplots(
        nrows=data_td.shape[0],
        ncols=is_draw_td + is_draw_fd,
        squeeze=False,
        sharex="col",
        sharey="col" if is_share_y else False,
    )
    for ch in range(data_td.shape[0]):
        if is_draw_td:
            # # # plot IR # # #
            length = len(data_td[ch])
            if length > _TD_STEM_LIM:
                axes[ch, td_col].plot(
                    np.arange(0, length), data_td[ch], linewidth=0.5, color="C0"
                )
            else:
                axes[ch, td_col].stem(
                    data_td[ch],
                    linefmt="C0-",
                    markerfmt="C0.",
                    basefmt="C0-",
                    use_line_collection=True,
                )
            # set limits
            if is_etc:
                # needs to be done before setting yticks
                axes[ch, td_col].set_ylim(*_adjust_y_lim(is_fd=False))
            # set ticks and grid
            axes[ch, td_col].tick_params(
                which="major",
                direction="in",
                top=True,
                bottom=True,
                left=True,
                right=True,
            )
            axes[ch, td_col].tick_params(which="minor", length=0)
            if is_draw_grid:
                from . import config

                length_2 = 2 ** np.ceil(np.log2(length))  # next power of 2
                if length > config.BLOCK_LENGTH:
                    axes[ch, td_col].set_xticks(
                        np.arange(
                            0,
                            length + 1,
                            length_2 // 4 if length_2 > length else length // 2,
                        ),
                        minor=False,
                    )
                    axes[ch, td_col].set_xticks(
                        np.arange(0, length + 1, config.BLOCK_LENGTH), minor=True
                    )
                    axes[ch, td_col].grid(
                        True, which="both", axis="x", color="r", alpha=0.4
                    )
                else:
                    axes[ch, td_col].set_xticks(
                        np.arange(0, length + 1, length_2 // 4 if length_2 > 4 else 2),
                        minor=False,
                    )
                    axes[ch, td_col].grid(True, which="major", axis="x", alpha=0.25)
                axes[ch, td_col].grid(True, which="both", axis="y", alpha=0.1)
                if is_etc:
                    axes[ch, td_col].set_yticks(
                        np.arange(*axes[ch, td_col].get_ylim(), step_db_y), minor=True
                    )
                else:
                    axes[ch, td_col].axhline(y=0, color="black", linewidth=0.75)
            # set limits
            if length > _TD_STEM_LIM:
                axes[ch, td_col].set_xlim(0, length)
            else:
                # overwrite ticks
                axes[ch, td_col].set_xticks(np.arange(0, length, 1), minor=False)
                axes[ch, td_col].set_xlim(-0.5, length - 0.5)
            # set labels
            if is_label_x and ch == data_td.shape[0] - 1:
                axes[ch, td_col].set_xlabel("Samples")
            # set axes in background
            axes[ch, td_col].set_zorder(-1)

        if is_draw_fd:
            # # # plot spectrum # # #
            axes[ch, fd_col].semilogx(
                np.linspace(0, fs / 2, len(data_fd[ch])),
                20 * np.log10(np.abs(data_fd[ch])),
                color="C1",
            )
            # ignore underflow FloatingPointError in `numpy.ma.power()`
            with np.errstate(under="ignore"):
                # set limits, needs to be done before setting yticks
                axes[ch, fd_col].set_ylim(*_adjust_y_lim(is_fd=True))
            # set ticks and grid
            axes[ch, fd_col].set_xticks(_FREQS_LABELED)
            axes[ch, fd_col].xaxis.set_major_formatter(
                FuncFormatter(lambda x, _: f"{x / 1000:.16g}")
            )
            axes[ch, fd_col].tick_params(
                which="major",
                direction="in",
                top=True,
                bottom=True,
                left=True,
                right=True,
            )
            axes[ch, fd_col].tick_params(which="minor", length=0)
            if is_draw_grid:
                axes[ch, fd_col].grid(True, which="major", axis="both", alpha=0.25)
                axes[ch, fd_col].grid(True, which="minor", axis="both", alpha=0.1)
                axes[ch, fd_col].set_yticks(
                    np.arange(*axes[ch, fd_col].get_ylim(), step_db_y), minor=True
                )
            # set limits, needs to be done after setting xticks
            axes[ch, fd_col].set_xlim(*set_fd_f_x)
            # set labels
            if is_label_x and ch == data_td.shape[0] - 1:
                axes[ch, fd_col].set_xlabel("Frequency / kHz")
            # set axes in background
            axes[ch, fd_col].set_zorder(-1)

        # set legend
        if (is_draw_td or is_draw_fd) and lgd_ch_ids:
            lgd_str = f'ch {lgd_ch_ids[ch]}{" (ETC)" if is_etc and is_draw_td else ""}'
            axes[ch, td_col].legend(
                labels=[lgd_str], loc="upper right", fontsize="xx-small"
            )

    # remove layout margins
    fig.tight_layout(pad=0)

    if is_show_blocked is not None:
        plt.show(block=is_show_blocked)

    return fig


def plot_nm_rms(
    data_nm_fd,
    lgd_ch_ids=None,
    bar_width=0.9,
    min_dr_db_y=20,
    step_db_y=1.0,
    is_show_blocked=None,
):
    """
    Parameters
    ----------
    data_nm_fd : numpy.ndarray
        spherical harmonics coefficients frequency domain data that should be plotted of size
        [number of blocks; number of SH order / modes; number of channels; number of bins]
    lgd_ch_ids : array_like, optional
        IDs that should be printed in the legend as individual channel names of size
        [number of channels] (range from 0 if nothing is specified)
    bar_width : float, optional
        width of individual bars relative to tick label distance
    min_dr_db_y : float, optional
        minimum dynamic range of RMS axis in dB
    step_db_y : float, optional
        step size of RMS axis in dB for minor grid and rounding of limits
    is_show_blocked : bool, optional
        if figure should be shown with the provided `block` status

    Returns
    -------
    matplotlib.figure.Figure
        generated plot
    """

    def _neg_tick(rms, _):
        return f"{rms + base_rms if rms != base_rms else 0:.1f}"

    import numpy as np
    from sound_field_analysis import sph
    from matplotlib.ticker import FuncFormatter

    _MN_LABEL_FONT = {"family": "monospace", "size": "small"}

    # check provided data size
    data_nm_fd = np.atleast_3d(data_nm_fd)
    if data_nm_fd.ndim == 3:
        data_nm_fd = data_nm_fd[np.newaxis, :]
    elif data_nm_fd.ndim > 4:
        data_nm_fd = data_nm_fd.squeeze()
        if data_nm_fd.ndim > 4:
            raise ValueError(
                f"plotting of data with size {data_nm_fd.shape} is not supported."
            )
    blocks, coeffs, chs, bins = data_nm_fd.shape

    # check provided legend IDs
    if lgd_ch_ids is None:
        if chs > 1:
            lgd_ch_ids = range(chs)
    elif not isinstance(lgd_ch_ids, (list, range, np.ndarray)):
        raise TypeError(
            f"legend channel IDs of type {type(lgd_ch_ids)} are not supported."
        )
    elif len(lgd_ch_ids) != chs:
        raise ValueError(
            f"length of legend channel IDs ({len(lgd_ch_ids)}) does not match "
            f"the size of the data ({chs})."
        )

    if bins > 1:
        # transform into time domain
        data_nm_td = np.fft.irfft(data_nm_fd, axis=-1)
    else:
        # time domain factor is identical to frequency domain
        data_nm_td = np.abs(data_nm_fd)

    # calculate RMS energy level per SH degree
    data_nm_rms = calculate_rms(data_nm_td, is_level=True)

    # calculate RMS upper and lower limits
    base_rms = step_db_y * np.floor(np.nanmin(data_nm_rms) / step_db_y)
    max_rms = step_db_y * np.ceil(np.nanmax(data_nm_rms) / step_db_y)

    # extend RMS dynamic range in case provided minimum value is not reached
    if max_rms - base_rms < abs(min_dr_db_y):
        base_rms = max_rms - abs(min_dr_db_y)

    # add to limits in case RMS values are exactly at limit
    if base_rms == np.nanmin(data_nm_rms):
        base_rms -= step_db_y
    if max_rms == np.nanmax(data_nm_rms):
        max_rms += step_db_y

    # generate x data and labels
    x = np.arange(coeffs)
    m, n = sph.mnArrays(nMax=int(np.sqrt(coeffs) - 1))
    mn_str = [f'({n_i:d}, {f"{m_i:+d}" if m_i else " 0"})' for m_i, n_i in zip(m, n)]
    n_change = np.where(np.diff(n) > 0)[0]
    n_change = np.append(n_change, coeffs - 1)

    # generate figure
    formatter = FuncFormatter(_neg_tick)
    fig, axes = plt.subplots(
        nrows=chs, ncols=blocks, squeeze=False, sharex="all", sharey="all"
    )
    for ch in range(chs):
        for b in range(blocks):
            axes[ch, b].yaxis.set_major_formatter(formatter)

            # plot background span to distinguish SH orders
            for i in range(n_change.size):
                n_start = 0 if i == 0 else n_change[i - 1] + 1
                n_end = n_change[i]
                axes[ch, b].axvspan(
                    n_start - 0.5,
                    n_end + 0.5,
                    facecolor=f"{i / n_change.size:f}",
                    alpha=0.6,
                )

            # plot grouped RMS values
            bar = axes[ch, b].bar(
                x=x,
                height=data_nm_rms[b, :, ch] - base_rms,
                width=bar_width,
                align="center",
                color="C3",
                edgecolor="black",
                linewidth=0.5,
                zorder=3,
            )

            # set axis labels and ticks
            axes[ch, b].grid(True, which="major", axis="y", alpha=1.0, zorder=0)
            axes[ch, b].grid(True, which="minor", axis="y", alpha=0.4, zorder=0)
            axes[ch, b].tick_params(
                which="major",
                direction="in",
                top=True,
                bottom=True,
                left=True,
                right=True,
            )
            axes[ch, b].tick_params(which="minor", length=0)
            axes[ch, b].set_xticks(x)
            axes[ch, b].set_yticks(
                np.arange(*axes[ch, b].get_ylim(), step_db_y), minor=True
            )
            if b == 0:
                axes[ch, b].set_ylabel("RMS in dB")
            if ch == chs - 1:
                axes[ch, b].set_xlabel("SH (order, degree)")
                axes[ch, b].set_xticklabels(
                    mn_str, rotation=90, fontdict=_MN_LABEL_FONT
                )

            # set legend
            if lgd_ch_ids:
                axes[ch, b].legend(
                    handles=[bar],
                    labels=[f"ch {lgd_ch_ids[ch]}"],
                    loc="upper right",
                    fontsize="xx-small",
                )

    # set axis limits
    plt.xlim(-0.5, coeffs - 0.5)
    plt.ylim(0, max_rms - base_rms)  # according to base RMS label offset

    # remove layout margins
    fig.tight_layout(pad=0)

    if is_show_blocked is not None:
        plt.show(block=is_show_blocked)

    return fig


def export_plot(figure, name, logger=None, file_type="png"):
    """
    Parameters
    ----------
    figure : matplotlib.figure.Figure
        plot that should be exported
    name : str
        name or path of image file being exported, in case no path is given standard logging
        directory will be used
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
    for ending in re.split(r"[,.;:|/\-+]+", file_type):
        file = f"{name}{os.path.extsep}{ending}"
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
    html_all = """
<html lang="en">
    <head>
        <title></title>
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
        """

    # add title
    if title:
        html_all += f"<h2> {title} </h2>\n"

    # add content
    html_all += html_content

    # add footer
    html_all += """
    </body>
</html>
"""

    # write file
    with open(file_name, mode="w") as f:
        f.write(html_all)


SPEED_OF_SOUND = 343
"""Speed of sound in meters per second in air."""
SEPARATOR = "-------------------------------------------------------------------------"
"""String to improve visual orientation for a clear logging behaviour."""
_RNG = _set_noise_generator()
"""Random number generator instance to be reused during rendering for best real-time performance."""
