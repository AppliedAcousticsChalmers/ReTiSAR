import logging
import os
import sys

from . import config


class LoggerLevelFilter(logging.Filter):
    """
    Mimic the standard logging behaviour, so only logging entries of level `logging.WARNING` or above are shown in
    `sys.stderr`.
    """

    def filter(self, record):
        """
        Overrides the `logging.Filter` default function.

        Parameters
        ----------
        record : logging.LogRecord
            current logging record

        Returns
        -------
        bool
            if record should be caught by filter (not shown in log)
        """
        return record.levelno < logging.WARNING


_LOGGING_REPLACER = '@'
"""Character to replace in logging related configuration strings from `config`, see `config.LOGGING_FORMAT`."""


def setup(subprocess_name=None, is_disable_file_logger=False):
    """
    Create a new logger instance with the given name. The logging format is set according to the options given in
    `config`. This is useful for getting convenient logging information if subprocesses are used in an application.

    With the help of `LoggerLevelFilter` two handlers will be implemented which mimic the default logging behaviour by
    outputting messages of level `logging.WARNING` to `sys.stderr` and everything below to `sys.stdout`.

    Parameters
    ----------
    subprocess_name : str, optional
        give name in case you want to create a logger for a subprocess
    is_disable_file_logger : bool, optional
        disable creation of and output into a log file

    Returns
    -------
    logging.Logger
        created instance, which should be used in a way like `logger.warning()`
    """
    logger = logging.getLogger(subprocess_name)
    logger.propagate = False  # disable propagation to parent log

    fmt = _get_format_with_extending_name(len(logger.name))
    logger = _update_format_recursively(logger, fmt)

    if logger.hasHandlers():
        logger.debug(f'created LOGGER "{logger.name}" but had inherited handlers already.')
    else:
        # general acceptance level
        logger.setLevel(config.LOGGING_LEVEL)

        # setup handler for output to sys.stdout
        # with filter to not duplicate messages shown in sys.stderr
        handler_std = logging.StreamHandler(sys.stdout)
        handler_std.setFormatter(logging.Formatter(fmt))
        handler_std.setLevel(config.LOGGING_LEVEL)
        handler_std.addFilter(LoggerLevelFilter())
        logger.addHandler(handler_std)

        # setup handler for output to sys.stderr
        handler_err = logging.StreamHandler(sys.stderr)
        handler_err.setFormatter(logging.Formatter(fmt))
        handler_err.setLevel(logging.WARNING)
        logger.addHandler(handler_err)

        if not is_disable_file_logger:
            logger.file = setup_logfile(logger.name)
            if logger.file:
                # setup handler for output to logfile
                handler_file = logging.FileHandler(logger.file, 'a')
                handler_file.setFormatter(logging.Formatter(f'%(asctime)s  {fmt}'))
                handler_file.setLevel(logging.NOTSET)
                logger.addHandler(handler_file)
                logger.debug(f'created LOGGER file "{os.path.relpath(logger.file)}".')

        logger.debug(f'created LOGGER "{logger.name}" with new handlers.')

    logger.debug(f'initialized LOGGER [{logger.name}] at level [{logging.getLevelName(logger.getEffectiveLevel())}].')
    return logger


def setup_logfile(name, ending='log'):
    """
    Prepares the infrastructure of logging to a file, in case there was a logfile name given in `config`. Potentially
    already existing files from a former session get backed up with an altered name (former backups get overwritten
    during that).

    Parameters
    ----------
    name : str
        file name (given by `logging.logger` instance or otherwise)
    ending : str, optional
        file ending of generated file, a default value of "log" is used

    Returns
    -------
    str
        relative path to the generated file
    """
    # get file name and path
    try:
        file = os.path.abspath(config.LOGGING_PATH)
    except (AttributeError, TypeError):
        return None  # no logging file name given

    # append file ending
    file = f'{os.path.join(file, name)}.{ending}'

    # create path if does not exist
    if not os.path.exists(os.path.dirname(file)):
        os.makedirs(os.path.dirname(file))

    # rename existing files as backup
    if os.path.isfile(file):
        backup = os.path.join(os.path.dirname(file), f'BACKUP_{os.path.basename(file)}')
        os.rename(file, backup)

    return file


def _update_format_recursively(logger, fmt):
    """
    Update `logging.Formatter` style for every contained `logging.Handler` by `_update_format_all_handlers()` as well
    as the parent log. This is done in a recursive way up to the root log. From the root log all existing loggers are
    updated in the same way by `_update_format_for_children()`.

    Parameters
    ----------
    logger : logging.Logger
        original instance
    fmt : str
        logging.Formatter style string

    Returns
    -------
    logging.Logger
        recursively updated instance
    """
    _update_format_all_handlers(logger, fmt)

    if logger.parent:
        logger.parent = _update_format_recursively(logger.parent, fmt)
    else:
        _update_format_for_children(logger, fmt)

    return logger


def _update_format_all_handlers(logger, fmt):
    """
    Update style for every contained `logging.Handler` of one `logging.Logger`.

    Parameters
    ----------
    logger : logging.Logger
        current instance
    fmt : str
        logging.Formatter style string
    """
    for handler in logger.handlers:  # type: logging.Handler
        # (always creating a new formatter with updated settings does not seem reasonable)
        handler.formatter._fmt = fmt
        # noinspection PyProtectedMember
        handler.formatter._style._fmt = fmt


def _update_format_for_children(logger, fmt):
    """
    Update style for every existing `logging.Logger` in `logger.manager.loggerDict` (if not propagating).

    Parameters
    ----------
    logger : logging.Logger
        root instance
    fmt : str
        logging.Formatter style string
    """
    for name in logger.manager.loggerDict:
        logger_child = logging.getLogger(name)
        if not logger_child.propagate:
            _update_format_all_handlers(logger_child, fmt)


def _get_format_with_extending_name(current_length):
    """
    Get logging format string as set in `config`. In case it contains a placeholder for self-extending name length (see
    `config._LOGGING_SHOW_REPLACER`), it will be replaced by the highest length of all logger names so far.

    Parameters
    ----------
    current_length : int
        length of name of current generated logger

    Returns
    -------
    str
        final logging format string
    """
    fmt = config.LOGGING_FORMAT
    if _LOGGING_REPLACER in fmt:
        max_length = _update_and_get_max_name_length(current_length)
        fmt = fmt.replace(_LOGGING_REPLACER, str(max_length))

    return fmt


def _update_and_get_max_name_length(current_length):
    """
    Updates (or creates in case it was not existing) a parameter in `config`. It is storing the length of the longest
    logger name created so far.

    Parameters
    ----------
    current_length : int
        current length of name

    Returns
    -------
    int
        maximum length to fit all names
    """
    try:
        # noinspection PyUnresolvedReferences
        max_length = config.logging_show_name_length
    except AttributeError:
        max_length = 0  # attribute was not existing before

    if current_length > max_length:
        config.logging_show_name_length = current_length

    # noinspection PyUnresolvedReferences
    return config.logging_show_name_length
