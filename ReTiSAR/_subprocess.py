import logging

from . import config, mp_context, process_logger, tools


class SubProcess(mp_context.Process):
    """
    Base functionality to run arbitrary functionality in a separate process. To run the process the functions `start()`,
    `join()` and `terminate()` have to be used.

    Attributes
    ----------
    _logger : logging.Logger
        logger used to distribute logging messages in a convenient manner
    _event_terminate : multiprocessing.Event
        event handling a thread safe flag to keep the process alive as long as it is not set
    _osc_client : pythonosc.udp_client.SimpleUDPClient
        OSC client instance used to send status messages
    _osc_name : str
        used OSC target when sending messages
    """

    def __init__(self, name, is_daemon_process=False, is_disable_file_logger=False, is_disable_logger=False):
        """
        Create a new instance of a separate process. According to the official documentation all attributes have be
        initialized in this `__init()__` function, to be available after running the spawned process.

        Parameters
        ----------
        name : str
            name of the spawned process
        is_daemon_process : bool, optional
            run as a daemon process (see documentation)
        is_disable_file_logger : bool, optional
            disable creation of and output into a log file, useful for benchmarking
        is_disable_logger : bool, optional
            disable logger after creation, useful for benchmarking
        """
        # self._logger.debug(f'initializing PROCESS [{name}] ...')
        mp_context.Process.__init__(self, name=name)
        self.daemon = is_daemon_process

        # initialize attributes
        self._osc_client = None
        self._osc_name = None

        # initialize logger
        self._logger = process_logger.setup(self.name, is_disable_file_logger)
        if is_disable_logger:
            self._logger.disabled = True

        # initialize attributes
        self._event_terminate = mp_context.Event()

    def _init_osc_client(self):
        """Initialize OSC client specific attributes to open port sending status data."""
        if not config.REMOTE_OSC_PORT:
            return

        from pythonosc import udp_client

        address = '127.0.0.1'
        port = config.REMOTE_OSC_PORT + 1
        self._osc_client = udp_client.SimpleUDPClient(address, port)

        self._osc_name = tools.transform_into_osc_target(self.name)
        self._logger.debug(f'sending OSC messages at ({address}, {port}, {self._osc_name}) ...')
        # actual OSC messages are generated in `process()` or adjacent functions in case port was opened

    def start(self):
        """
        Extends the `multiprocessing.Process` function to `start()` the process. This function only activates the JACK
        client and sets the ready event if it isn't called via a `super()` call. Inheriting classes overriding this
        function need to do these actions on their own.
        """
        self._logger.debug('starting PROCESS ...')
        super().start()

    def run(self):
        """
        Overrides the `multiprocessing.Process` function which is automatically called by `start()` to run the process.
        This implementation generates basic logging messages and makes the process stay alive until the according
        `multiprocessing.Event` is set.
        """
        self._logger.debug('running PROCESS ...')

        try:
            self._event_terminate.wait()
        except KeyboardInterrupt:
            self._logger.error('interrupted by user.')

    def terminate(self):
        """
        Extends the `multiprocessing.Process` function to terminate the process and generate some logging messages.
        """
        self._logger.debug('terminating PROCESS ...')
        self._event_terminate.set()

        # delete logger instance
        try:
            # noinspection PyUnresolvedReferences
            del self._logger.root.manager.loggerDict[self._logger.name]
        except KeyError:
            pass
        super().terminate()
