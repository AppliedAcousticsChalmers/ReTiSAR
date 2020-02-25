import logging
from sys import platform
from time import sleep

import jack
import numpy as np

from . import config, mp_context, tools
from ._convolver import DelayBuffer
from ._subprocess import SubProcess


class JackClient(SubProcess):
    """
    Base functionality to run a JACK client contained in a separate process. To run the process
    the functions `start()`, `join()` and `terminate()` have to be used.

    Attributes
    ----------
    _event_ready : multiprocessing.Event
        event handling a thread safe flag to indicate if the JACK client is initialized entirely
    _counter_dropout : multiprocessing.Value
        number of processed audio frames where the JACK client produced an `xrun`
    _client : jack.Client
        contained JACK client (see JACK-Client for documentation)
    _is_main_client : bool
        if logging messages about other clients should be shown by this client
    _is_detect_clipping : bool
        if output ports should be monitored to detect clipping
    _is_measure_levels : bool
        if output ports should be monitored to for its RMS level
    _is_measure_load : bool
        if client should report individual client and overall JACK system load
    _is_single_precision : bool
        if processing should be done based on `float32`, `float64` (double precision) otherwise
    _is_first_frame : bool
        if first frame is being processed to check array structures
    _output_mute : bool
        if client is suppressing all audio output
    _output_volume : float
        absolute multiplier as output volume scaling, applied before writing samples into
        `_client.outports.get_array()`
    _output_volume_relative : numpy.ndarray
        relative individual multiplier as output volume scaling per channel, applied on top of
        absolute output volume multiplier
    _input_buffer : DelayBuffer
        buffer realizing an adjustable delay line before the input signals are delivered to
        internal processing
    """

    def __init__(
        self,
        name,
        block_length=None,
        output_volume_dbfs=0,
        input_delay_ms=0,
        is_single_precision=True,
        is_main_client=True,
        is_detect_clipping=True,
        is_measure_levels=False,
        is_measure_load=True,
        *args,
        **kwargs,
    ):
        """
        Create a new instance of a JACK client in a separate process. Any of the provided JACK
        callbacks (see documentation) can be individually overridden. Here this is mainly used to
        generate meaningful logging messages, hence the implementations are not further documented.

        Parameters
        ----------
        name : str
            name of the JACK client and spawned process
        block_length : int, optional
            block length in samples of the JACK client (global setting for JACK server and all
            clients)
        output_volume_dbfs : float or int, optional
            starting output volume of the client in Decibel_FullScale
        input_delay_ms : float, optional
            starting input buffer delay of the client in milliseconds
        is_single_precision : bool, optional
            if processing should be done based on `float32`, `float64` (double precision) otherwise
        is_main_client : bool, optional
            if logging messages about other clients should be shown by this client
        is_detect_clipping : bool, optional
            if output ports should be monitored to detect clipping
        is_measure_levels : bool, optional
            if output ports should be monitored to for its RMS level
        is_measure_load : bool, optional
            if client should report individual client and overall JACK system load
        """
        super().__init__(name, *args, **kwargs)

        # initialize attributes
        self._is_main_client = is_main_client
        self._is_detect_clipping = is_detect_clipping
        self._is_measure_levels = is_measure_levels
        self._is_measure_load = is_measure_load
        self._is_single_precision = is_single_precision
        self._is_first_frame = True
        self._output_mute = False
        self._output_volume = pow(10, output_volume_dbfs / 20.0)
        self._output_volume_relative = None
        self._event_ready = mp_context.Event()
        self._counter_dropout = mp_context.Value("i")

        self._init_client(block_length)
        self._input_buffer = DelayBuffer(
            sample_rate=self._client.samplerate,
            block_length=block_length,
            delay_ms=input_delay_ms,
        )  # needs to be done after initializing the client

        if self._is_main_client:
            self._logger.debug("setting general JACK callbacks ...")

            @jack.set_error_function
            def error(msg):
                self._logger.error(f"[JACK] {msg}")
                self._event_terminate.set()

            @jack.set_info_function
            def info(msg):
                self._logger.info(f"[JACK] {msg}")

        # setting JACK callbacks
        self._logger.debug("setting JACK client callbacks ...")

        # noinspection PyUnusedLocal
        @self._client.set_process_callback
        def process(frames):
            """
            Process signals block wise. It should not be necessary to override this function by
            deriving classes. Rather modify the individual implementations in
            `_process_receive()`, `_process()` and `_process_deliver()`.

            Parameters
            ----------
            frames : int
                number of samples in the current block
            """
            if self._event_terminate.is_set() or not self._client.outports:
                return

            # this should not be necessary here, but prevents errors when restarting `JackPlayer`
            self._event_ready.wait()

            # receive, process and deliver audio blocks
            self._process_deliver(self._process(self._process_receive()))

            # measure and log / provide individual client and overall system load
            self._report_load()

        @self._client.set_shutdown_callback
        def shutdown(status, reason):
            self._logger.warning(
                f'shutting down JACK status {status}, reason "{reason}".'
            )
            self._event_terminate.set()

        @self._client.set_freewheel_callback
        def freewheel(starting):
            self._logger.info(
                f'{["stopping", "starting"][starting]} JACK freewheel mode.'
            )

        # noinspection PyShadowingNames
        @self._client.set_blocksize_callback
        def blocksize(blocksize):
            if self._event_ready.is_set() and blocksize != self._client.blocksize:
                self._logger.error(
                    "change of JACK block size detected while being active."
                )
            else:
                lvl = logging.INFO if self._is_main_client else logging.DEBUG
                self._logger.log(lvl, f"setting JACK blocksize to {blocksize}.")

        # noinspection PyShadowingNames
        @self._client.set_samplerate_callback
        def samplerate(samplerate):
            lvl = logging.INFO if self._is_main_client else logging.DEBUG
            self._logger.log(lvl, f"JACK samplerate was set to {samplerate}.")

        # noinspection PyShadowingNames
        @self._client.set_client_registration_callback
        def client_registration(name, register):
            if self._is_main_client:  # only show by one client
                self._logger.debug(
                    f'{["unregistered", "registered"][register]} JACK client {name}.'
                )

        @self._client.set_port_registration_callback
        def port_registration(port, register):
            if isinstance(port, jack.OwnPort):  # only show for own ports
                self._logger.debug(
                    f'{["unregistered", "registered"][register]} JACK port {port}.'
                )

        @self._client.set_port_connect_callback
        def port_connect(a, b, connect):
            if isinstance(a, jack.OwnPort):  # only show if client is the sending unit
                self._logger.debug(
                    f'{["disconnected", "connected"][connect]} JACK {a} and {b}.'
                )

        try:

            @self._client.set_port_rename_callback
            def port_rename(port, old, new):
                if isinstance(port, jack.OwnPort):  # only show for own ports
                    self._logger.debug(
                        f'renamed JACK port {port} from "{old}" to "{new}".'
                    )

        except AttributeError:
            self._logger.warning(
                "Could not register JACK port rename callback (not available on JACK1)."
            )

        @self._client.set_xrun_callback
        def xrun(delay):
            if (
                not self._event_ready.is_set()
                or not config.IS_RUNNING.is_set()
                or self._output_mute
            ):
                return
            # if delay > 0 and self._is_main_client:  # only show for main client
            if delay > 0:  # show for each client
                lvl = logging.DEBUG if config.IS_DEBUG_MODE else logging.WARNING
                self._logger.log(
                    lvl, f"occurred JACK xrun (delay {delay} microseconds)."
                )
            with self._counter_dropout.get_lock():
                self._counter_dropout.value += 1

        # setting up OSC sender
        self._init_osc_client()

    def _init_client(self, block_length):
        """
        Initialize JACK client specific attributes.

        Parameters ---------- block_length : int block length in samples of the JACK client,
        must be a power of 2. Since this is a global setting for the JACK server, the last
        setting always gets applied. This should only be called before having any clients active,
        since the processing flow will be interrupted.

        Raises
        ------
        ValueError
            in case invalid block length is given
        """
        self._logger.info("initializing JACK client ...")

        # check for valid name length on OSX
        _OSX_MAX_SEMAPHORE_LENGTH = (
            30  # no idea where this (stupidly low) value comes from :(
        )
        _OSX_MAX_SEMAPHORE_LENGTH -= (
            3  # this comes from the added 'js_' as a prefix to every name
        )
        if platform == "darwin" and len(self.name) >= _OSX_MAX_SEMAPHORE_LENGTH:
            self._logger.warning(
                f"name with {len(self.name)} signs is too long on OSX (limit is "
                f"{_OSX_MAX_SEMAPHORE_LENGTH})."
            )
            self.name = self.name[len(self.name) - _OSX_MAX_SEMAPHORE_LENGTH :]
            self._logger.warning(f'name got shortened to "{self.name}".')

        self._client = jack.Client(self.name)
        if block_length:
            if not bool(block_length and not (block_length & (block_length - 1))):
                self._logger.error(
                    f"provided blocksize of {block_length} is not a power of 2."
                )
                raise ValueError(f'failed to create "{self.name}" instance.')

            self._client.blocksize = block_length

        if self._is_main_client:
            if self._client.status.server_started:
                self._logger.warning(
                    "[INFO]  JACK server was started for this application."
                )
            else:
                self._logger.warning("[INFO]  JACK server was already running.")
        if self._client.status.name_not_unique:
            self._logger.warning(
                f"assigned unique name to JACK client [{self._client.name!r}]."
            )

    def start(self):
        """
        Extends the `multiprocessing.Process` function to `start()` the process. This function
        only activates the JACK client and sets the ready event if it isn't called via a
        `super()` call. Inheriting classes overriding this function need to do these actions on
        their own.
        """
        super().start()

        # prevent setting _event_ready if called by an overridden function
        if type(self) is JackClient:  # do not replace with `isinstance()`
            self._logger.info("activating JACK client ...")
            self._client.activate()
            self._event_ready.set()

    def run(self):
        """
        Overrides the `multiprocessing.Process` function which is automatically called by
        `start()` to run the process. This implementation generates basic logging messages and
        makes the process stay alive until the according `multiprocessing.Event` is set.
        """
        self._logger.debug("waiting to run JACK client ...")
        self._event_ready.wait()
        self._logger.debug("running JACK client ...")

        super().run()

    def terminate(self):
        """
        Extends the `multiprocessing.Process` function to terminate the process, the JACK client
        and generate some logging messages.
        """
        if self._event_ready.is_set():
            self.terminate_members()

        self._logger.info("terminating JACK client ...")
        self._client.deactivate()
        self._client.close()
        super().terminate()

    def terminate_members(self):
        """
        Terminate the JACK client elements of this process. This function is separated from
        regular terminate(), so it can also be called by the process itself.
        """
        self._event_terminate.set()
        self._event_ready.clear()
        # needed to not crash when terminating clients i.e. at block_len=64 filter_len=110250
        sleep(0.05)
        self._client.deactivate()  # this is likely to not succeed if called by instance itself

    def get_client_outputs(self):
        """
        Returns
        -------
        jack.Ports
            JACK client outports
        """
        return self._client.outports

    # noinspection DuplicatedCode
    def _client_register_inputs(self, input_count):
        """
        Parameters
        ----------
        input_count : int
            number of input ports to be registered to the current client

        Raises
        ------
        RuntimeError
            re-raise of jack.JackError
        """
        if input_count <= len(self._client.inports):
            return

        # cleanup existing input ports
        self._client.inports.clear()

        try:
            # create input ports according to source channel number (index starting from 1)
            for number in range(1, input_count + 1):
                self._client.inports.register(f"input_{number}")
        except jack.JackError as e:
            raise RuntimeError(f"[JackError]  {e}")

        # recreate input buffer
        self._input_buffer.init(
            max_length_sec=config.CLIENT_MAX_DELAY_SEC,
            channel_count=len(self._client.inports),
        )

    def client_register_and_connect_inputs(self, source_ports=True):
        """
        Register an identical number of input ports according to the provided source ports to the
        current client. Afterwards connect the given ports to the newly created target ports in a
        1:1 relation.

        Parameters
        ----------
        source_ports : jack.Ports or bool or None, optional
            source ports for connecting the created input ports to. If None is given, a port number
            according to the system recording ports is registered. If True is given, ports will
            be registered and also connected to the system recording ports. If False is given,
            no ports will be registered or connected.

        Raises
        ------
        ValueError
            in case no source ports are provided and also no physical recording ports are found
        """
        if source_ports is False:
            return

        is_connect = source_ports is not None

        # temporarily pause execution
        event_ready_state_before = self._event_ready.is_set()
        self._event_ready.clear()

        if type(source_ports) is not jack.Ports:
            # get physical recording ports in case no ports were given
            source_ports = self._client.get_ports(is_physical=True, is_output=True)
            if not source_ports:
                raise ValueError(
                    "no source ports given and no physical recording ports detected."
                )

        self._client_register_inputs(len(source_ports))

        if is_connect:
            # connect source to input ports
            for src, dst in zip(source_ports, self._client.inports):
                self._client.connect(src, dst)

        # restore beforehand execution state
        if event_ready_state_before:
            self._event_ready.set()

    # noinspection DuplicatedCode
    def _client_register_outputs(self, output_count):
        """
        Parameters
        ----------
        output_count : int
            number of output ports to be registered to the current client

        Raises
        ------
        RuntimeError
            re-raise of jack.JackError
        """
        if output_count <= len(self._client.outports):
            return

        # cleanup existing output ports
        self._client.outports.clear()

        try:
            # create output ports (index starting from 1)
            for number in range(1, output_count + 1):
                self._client.outports.register(f"output_{number}")
        except jack.JackError as e:
            raise RuntimeError(f"[JackError]  {e}")

        # adjust size of relative output volume multiplier
        self._output_volume_relative = np.ones(
            shape=(output_count, 1), dtype=np.float32
        )

    def _client_register_and_connect_outputs(self, target_ports=True):
        """
        Register an identical number of output ports according to the provided target ports to
        the current client. Afterwards connect the given ports to the newly created source ports
        in a 1:1 relation.

        Parameters
        ----------
        target_ports : jack.Ports or bool or None, optional
            target ports for connecting the created output ports to. If None is given, a port
            number according to the system playback ports is registered. If True is given,
            ports will be registered and also connected to the system playback ports. If False is
            given, no ports will be registered or connected.

        Raises
        ------
        ValueError
            in case no target ports are provided and also no physical playback ports are found
        """
        if target_ports is False:
            return

        is_connect = target_ports is not None

        # temporarily pause execution
        event_ready_state_before = self._event_ready.is_set()
        self._event_ready.clear()

        if not len(self._client.outports):
            # register output ports if none exist
            self._client_register_outputs(2)

        if type(target_ports) is not jack.Ports:
            # get physical playback ports in case no ports were given
            target_ports = self._client.get_ports(is_physical=True, is_input=True)
            if not target_ports:
                raise ValueError(
                    "no target ports given and no physical playback ports detected."
                )

        if is_connect:
            # connect output to target ports
            for source, target in zip(self._client.outports, target_ports):
                self._client.connect(source, target)

        # restore beforehand execution state
        if event_ready_state_before:
            self._event_ready.set()

    def set_input_delay_ms(self, new_delay=0.0):
        """
        Parameters
        ----------
        new_delay : float, optional
            new input buffer delay of the client in milliseconds, if no value is given the delay is
            set to 0

        Returns
        -------
        float
            actually realized input buffer delay in milliseconds
        """
        if not self._check_alive("set delay"):
            return

        new_delay = self._input_buffer.set_delay(new_delay_ms=new_delay)
        self._logger.info(f"set delay to {new_delay:.1f} ms.")
        return new_delay

    def set_output_mute(self, new_state=None):
        """
        Parameters
        ----------
        new_state : bool, int, float, str or None, optional
            new output mute state of the client, if `None` is given this function works as a toggle
            between states

        Returns
        -------
        bool
            actually realized output mute state
        """
        if not self._check_alive("set mute state"):
            return

        new_state = tools.transform_into_state(new_state, logger=self._logger)
        # toggle state
        if new_state is None:
            new_state = not self._output_mute

        if self._output_mute != new_state:
            self._output_mute = new_state
            self._logger.info(f'set mute state to {["OFF", "ON"][self._output_mute]}.')

        return new_state

    def set_output_volume_db(self, value_db_fs=0.0):
        """
        Parameters
        ----------
        value_db_fs : float or int, optional
            new global output volume of the client in Decibel_FullScale

        Returns
        -------
        float
            actually realized global output volume in Decibel_FullScale
        """
        log_str = "set output volume"
        if value_db_fs is None or not self._check_alive(log_str):
            return

        return self._set_output_volume(
            log_str=log_str, value_db_fs=value_db_fs, port=None
        )

    def set_output_volume_relative_db(self, value_db_fs=0.0):
        """
        Parameters
        ----------
        value_db_fs : float, None, list of float or numpy.ndarray, optional
            new relative individual output volumes of the client in Decibel_FullScale of size
            [number of output channels]

        Returns
        -------
        numpy.ndarray
            actually realized relative output volumes in Decibel_FullScale of size
            [number of output channels]
        """
        log_str = "set relative output volumes"
        if value_db_fs is None or not self._check_alive(log_str):
            return

        # reset to have no relative difference
        if isinstance(value_db_fs, float) or isinstance(value_db_fs, int):
            value_db_fs = np.zeros_like(self._output_volume_relative)
            return self._set_output_volume(
                log_str=log_str, value_db_fs=value_db_fs, port=None
            )

        # always transform into numpy.ndarray
        if not isinstance(value_db_fs, np.ndarray):
            value_db_fs = np.array(value_db_fs, dtype=np.float32)

        # check number of dimensions
        value_db_fs = value_db_fs.squeeze()
        if value_db_fs.ndim > 1:
            self._logger.error(
                f"provided volume array has more than one dimension ({value_db_fs.ndim})"
                f', "{log_str}" ignored.'
            )
            return

        # check vector length
        if value_db_fs.size != len(self._client.outports):
            self._logger.error(
                f"provided volume array length ({value_db_fs.size}) does not match the number of "
                f'output channels ({len(self._client.outports)}), "{log_str}" ignored.'
            )
            return

        return self._set_output_volume(
            log_str=log_str, value_db_fs=value_db_fs[:, np.newaxis], port=None
        )

    def set_output_volume_port_relative_db(self, port, value_db_fs=0.0):
        """
        Parameters
        ----------
        port : int
            number of individual port of the client, value 0 .. number of ports - 1
        value_db_fs : float, optional
            new relative individual output volume of the port in Decibel_FullScale

        Returns
        -------
        float
            actually realized relative output volume of the port in Decibel_FullScale
        """
        # always cast to Integer
        port = int(port)

        log_str = f"set relative output volume of port {port}"
        if value_db_fs is None or not self._check_alive(log_str):
            return

        # check singular value is provided
        if isinstance(value_db_fs, list) or isinstance(value_db_fs, np.ndarray):
            if len(value_db_fs) > 1:
                self._logger.error(
                    f"provided volume has more than one value ({len(value_db_fs)}), "
                    f'"{log_str}" ignored.'
                )
                return
            # extract first element
            value_db_fs = value_db_fs[0]

        # check port number
        if not 0 <= port < len(self._client.outports):
            self._logger.error(
                f'provided port number is out of range, "{log_str}" ignored.'
            )
            return

        return self._set_output_volume(
            log_str=log_str, value_db_fs=value_db_fs, port=port
        )

    def _set_output_volume(self, log_str, value_db_fs, port):
        # convert magnitude into decibel
        output_volume = 10 ** (value_db_fs / 20.0)

        # set_output_volume_port_relative_db
        if (
            isinstance(port, int)
            and self._output_volume_relative[port] != output_volume
        ):
            self._output_volume_relative[port] = output_volume
            self._logger.info(f"{log_str} to {value_db_fs:+.1f} dBFS.")

        # set_output_volume_relative_db()
        elif (
            port is None
            and isinstance(output_volume, np.ndarray)
            and not np.equal(self._output_volume_relative, output_volume).all()
        ):
            self._output_volume_relative = output_volume
            val_str = np.array2string(
                value_db_fs[:, 0],
                max_line_width=100,
                separator=",",
                precision=1,
                floatmode="fixed",
                sign="+",
            )
            self._logger.info(f"{log_str} to {val_str} dBFS.")

        # set_output_volume_db()
        elif (
            port is None
            and isinstance(output_volume, float)
            and self._output_volume != output_volume
        ):
            self._output_volume = output_volume
            self._logger.info(f"{log_str} to {value_db_fs:+.1f} dBFS.")

        # print warning for positive dB_FS gain
        if np.greater(self._output_volume * self._output_volume_relative, 1).any():
            self._logger.warning("setting output volume > 0 dBFS.")

        return value_db_fs

    def _process_receive(self):
        """
        Gather input audio blocks from JACK. Optionally the memory structure of the data will be
        checked here.

        Returns
        -------
        numpy.ndarray
            block of audio data that was received from JACK

        Notes
        -----
        When receiving the input arrays from JACK it is necessary to store copies when data needs
        to persist longer then this processing frame. This applies here, since at least one block
        is buffered and shifted internally even for un-partitioned convolution. In the current
        implementation `np.vstack()` creates a copy of the data.
        """
        if not self._client.inports or not config.IS_RUNNING.is_set():
            return None

        # receive input from JACK, `np.vstack()` creates a copy
        input_td = np.vstack([port.get_array() for port in self._client.inports])

        # buffer and delay input
        input_td = self._input_buffer.process_block(input_td)

        # TODO: prevent all dynamic variable allocations
        # self._logger.info(id(input_td))

        # # check array structure
        # if not input_td.flags["C_CONTIGUOUS"]:
        #     self._logger.warning('input array not "C_CONTIGUOUS".')

        return input_td

    # noinspection PyMethodMayBeStatic
    def _process(self, input_td):
        """
        Process block of audio data. This implementation provides a straight passthrough
        behaviour. If actual signal processing should happen, a deriving class needs to override
        this function.

        Parameters
        ----------
        input_td : numpy.ndarray
            block of audio data that was received from JACK

        Returns
        -------
        numpy.ndarray
            processed block of audio data that will be delivered to JACK
        """
        # straight passthrough
        return input_td
        # TODO: skip processing in case output is muted or volume is 0?

    def _process_deliver(self, output_td):
        """
        Apply output volume to output audio blocks and deliver them to JACK. Optionally the
        memory structure and sample clipping of the data will be checked here.

        Output data channels greater then available output ports are neglected. Output ports
        greater then available output data are filled with zeros.

        Parameters
        ----------
        output_td : numpy.ndarray or None
            processed block of audio data that will be delivered to JACK
        """
        if self._event_terminate.is_set():
            return

        if self._output_mute or output_td is None:
            # output zeros
            for port in self._client.outports:
                port.get_array().fill(0)

        else:
            # apply output volume (scaling for all channels) as well as relative differences (
            # between channels)
            output_td *= self._output_volume * self._output_volume_relative

            if self._is_measure_levels:
                # calculate RMS level
                rms = tools.calculate_rms(output_td, is_level=True)
                if self._osc_client:
                    # OSC only works with float64, see
                    # https://github.com/attwad/python-osc/issues/102
                    self._osc_client.send_message(
                        f"{self._osc_name}/rms", np.round(rms, 1).astype(np.float64)
                    )
                else:
                    self._logger.info(
                        f"output RMS level ["
                        f'{np.array2string(rms, separator=",", precision=1, floatmode="fixed", sign="+")}]'
                    )

                # calculate PEAK level
                peak = tools.calculate_peak(output_td, is_level=True)
                if self._osc_client:
                    # OSC only works with float64, see
                    # https://github.com/attwad/python-osc/issues/102
                    self._osc_client.send_message(
                        f"{self._osc_name}/peak", np.round(peak, 2).astype(np.float64)
                    )
                else:
                    self._logger.debug(
                        f"output PEAK level ["
                        f'{np.array2string(peak, separator=",", precision=2, floatmode="fixed", sign="+")}]'
                    )

            # check array structure and dtype (first processing frame only)
            if self._is_first_frame:
                self._is_first_frame = False
                if output_td[0].flags["C_CONTIGUOUS"]:
                    self._logger.debug(f'output array layout is "C_CONTIGUOUS".')
                else:
                    self._logger.warning(f'output array layout is not "C_CONTIGUOUS".')
                if (self._is_single_precision and output_td.dtype == np.float32) or (
                    not self._is_single_precision and output_td.dtype == np.float64
                ):
                    self._logger.debug(
                        f'output array dtype is "{output_td.dtype}" as requested.'
                    )
                elif self._is_single_precision and output_td.dtype != np.float32:
                    self._logger.warning(
                        f'output array dtype is "{output_td.dtype}" instead of {np.float32}.'
                    )
                elif not self._is_single_precision and output_td.dtype != np.float64:
                    self._logger.warning(
                        f'output array dtype is "{output_td.dtype}" instead of {np.float64}.'
                    )
                else:
                    self._logger.warning(
                        f'output array dtype is unexpected: "{output_td.dtype}".'
                    )

            # regarding maximum number of ports or result channels
            for data, port in zip(output_td, self._client.outports):
                # check for clipping
                peak = np.abs(data).max()
                if self._is_detect_clipping and peak > 1:
                    self._logger.warning(
                        f"output clipping detected ({port.shortname} @ {peak:.2f})."
                    )

                # deliver output to JACK
                port.get_array()[:] = data  # assigning to a slice creates a copy

            # regarding ports greater then result channels
            for port in self._client.outports[output_td.shape[0] :]:
                # output zeros
                port.get_array().fill(0)

    def _report_load(self):
        """
        Deliver via OSC (or log otherwise) individual client load based on reported JACK frame
        times. Furthermore deliver (or log) overall system load based on a prediction by JACK.
        Latter is is done only when configured as a main client.
        """
        if not self._is_measure_load or not config.IS_RUNNING.is_set():
            return

        if self._is_main_client:
            # overall JACK system CPU load<r
            load = round(self.get_cpu_load(), 1)
            if self._osc_client:
                self._osc_client.send_message("/load", load)
            else:
                self._logger.debug(f"load system percent [{load}]")

        # individual client JACK system load
        t_percent = self._client.frame_time - self._client.last_frame_time
        t_percent = round(100 * t_percent / self._client.blocksize)
        if t_percent < 0:
            t_percent = 0

        if self._osc_client:
            self._osc_client.send_message(f"{self._osc_name}/load", t_percent)
        else:
            self._logger.debug(f"load percent [{t_percent}]")

    def _check_alive(self, msg=""):
        """
        Parameters
        ----------
        msg : str, optional
            specification of interrupted action as additional information in log message

        Returns
        -------
        bool
            if `SubProcess.is_alive()`, otherwise log error message
        """
        if self.is_alive():
            return True

        if not msg or msg == "":
            self._logger.error("client is not alive.")
        else:
            self._logger.error(f'client is not alive, "{msg}" ignored.')

        return False

    def get_cpu_load(self):
        """
        Returns
        -------
        float
             current CPU load estimated by JACK (would be reported identically by all clients)
        """
        return round(self._client.cpu_load(), 1)

    def get_dropout_counter(self):
        """
        This function is only useful for benchmarking purposes. Use the provided object in the
        following way to receive or reset its value in a multiprocess-safe manner:
            with counter.get_lock():
                 x = counter.value
                 counter.value = 0

        Returns
        -------
        multiprocessing.Value
             number of processed audio frames where the JACK client produced an `xrun`
        """
        return self._counter_dropout
