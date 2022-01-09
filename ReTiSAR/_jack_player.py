import logging
import os
import queue

import jack
import numpy as np
import soundfile

from . import config, mp_context
from ._jack_client import JackClient


class JackPlayer(JackClient):
    """
    Extended functionality from `JackClient` to also provide buffered playback of an audio file
    into the JACK output ports. To run the process the functions `start()`, `join()` and
    `terminate()` have to be used.

    Attributes
    ----------
    _event_play : multiprocessing.Event
        event handling a thread safe flag to indicate if the playback should be running
    _q : multiprocessing.Queue
        internal FIFO data buffer storing audio blocks being filled in `run()` and emptied within
        `process()`
    _sf : soundfile.Soundfile
        instance of the audio file to read
    _timeout : float
        controls the wait time in seconds, how long `run()` will wait at most until there is space
        in the buffer and also how often `process()` will have data in the buffer available
    _block_generator : generator
        generator allowing to block wise pull data from the audio file
    """

    def __init__(
        self, name, file_name, is_auto_play, buffer_length=20, *args, **kwargs
    ):
        """
        Extends the `JackClient` function to initialize a new JACK client and process. According
        to the documentation all attributes must be initialized in this function, to be available
        to the spawned process.

        Parameters
        ----------
        name : str
            name of the JACK client and spawned process
        file_name : str
            file path/name of audio file being played
        is_auto_play : bool
            if audio is supposed to be played after program start
        buffer_length : int, optional
            number of audio blocks (given by `jack.Client.blocksize`) being saved in the internal
            buffer
        """
        super().__init__(name=name, *args, **kwargs)

        # set attributes
        self._file_name = file_name
        self._buffer_length = buffer_length
        self._is_auto_play = is_auto_play

        # initialize attributes
        self._init_player()

        @self._client.set_port_registration_callback
        def port_registration(port, register):
            """
            Extends the basic logging behaviour by preventing to register any input ports to this
            class at all.

            Parameters
            ----------
            port : jack.Port
                current port being (un)registered
            register : bool
                if this is a register or unregister occurrence

            Raises
            ------
            NotImplementedError
                in case input port should be registered (not supported for this class)
            """
            if isinstance(port, jack.OwnPort):  # only show for own ports
                # prevent registration of input ports
                if register and port.is_input:
                    raise NotImplementedError(
                        "input ports to this class are not supported."
                    )

                # pay attention here to keep the same style as in JackClient...port_registration()
                if isinstance(port, jack.OwnPort):  # only show for own ports
                    self._logger.debug(
                        f'{["unregistered", "registered"][register]} JACK port {port}.'
                    )

    def _init_player(self):
        """
        Initialize audio player specific attributes.

        Raises
        ------
        ValueError
            in case buffer length is smaller than 1
        ValueError
            in case samplerate of input file does not match the JACK samplerate
        """
        if (
            not self._file_name
            or self._file_name.strip("'\"") == ""
            or self._file_name.upper().endswith("NONE")
        ):
            self._logger.warning("skipping audio source file playback.")
            # use `_counter_dropout` as indicator if file was loaded
            self._counter_dropout = None
            return

        if self._buffer_length < 1:
            self._logger.error(
                f"buffer length of {self._buffer_length} is smaller than 1."
            )
            raise ValueError(f'failed to create "{self.name}" instance.')

        self._event_play = mp_context.Event()
        self._q = mp_context.Queue(maxsize=self._buffer_length)

        # print file information
        file_info = (
            soundfile.info(self._file_name)
            .__str__()
            .split("\n", 1)[1]
            .replace("\n", ", ")
        )  # cut off name and reformat LF
        self._logger.info(
            f'opening file "{os.path.relpath(self._file_name)}"\n --> {file_info}'
        )
        self._sf = soundfile.SoundFile(file=self._file_name, mode="r")

        if self._sf.samplerate != self._client.samplerate:
            self._logger.error(
                f"input samplerate of {self._sf.samplerate} Hz does not match JACK samplerate of "
                f"{self._client.samplerate} Hz."
            )
            raise ValueError(f'failed to create "{self.name}" instance.')

        # create output ports according to file channel number
        self._client_register_outputs(self._sf.channels)

        # get matching numpy dtype according to provided file subtype. Currently
        # `soundfile.read()` only seems to have the types `float32`, `float64`, `int16` and
        # `int32` implemented, hence the decision is very easy
        dtype = np.float64 if self._sf.subtype.upper() == "DOUBLE" else np.float32
        # _SUBTYPE2DTYPE = {'PCM_16':  np.float16,
        #                   'PCM_24':  np.float32,
        #                   'PCM_32':  np.float32,
        #                   'FLOAT':   np.float32,
        #                   'DOUBLE':  np.float64,
        #                   'ALAC_16': np.float16,
        #                   'ALAC_20': np.float32,
        #                   'ALAC_24': np.float32,
        #                   'ALAC_32': np.float32, }
        # try:
        #     dtype = _SUBTYPE2DTYPE[self._sf.subtype]
        # except KeyError:
        #     raise NotImplementedError(
        #         f'numpy dtype according to subtype "{self._sf.subtype}" not yet implemented.'
        #     )

        if self._is_single_precision and dtype == np.float64:
            dtype = np.float32
        elif not self._is_single_precision and dtype == np.float32:
            self._logger.warning(
                f"[INFO]  file playback processed in single precision according to file subtype "
                f'"{self._sf.subtype}", even though double precision was requested.'
            )
            self._is_single_precision = True

        self._timeout = (
            self._client.blocksize * self._buffer_length / self._client.samplerate
        )
        self._block_generator = self._sf.blocks(
            blocksize=self._client.blocksize, dtype=dtype, always_2d=True, fill_value=0
        )

        # pre-fill queue
        self._q.put_nowait(
            np.zeros((self._sf.channels, self._client.blocksize), dtype=dtype)
        )

    def start(self):
        """
        Extends the `JackClient` function to `start()` the process. This is only necessary since
        the `super()` functions prevents activating the JACK client.
        """
        super().start()
        self._logger.debug("activating JACK client ...")
        self._client.activate()
        self._event_ready.set()

    def run(self):
        """
        Overrides the `JackClient` function to `run()` the process. This gathers data from the
        input audio file into the buffer, which is done in a block wise manner. None in the
        buffer is used to signal the end of the audio file.
        """
        self._logger.debug("waiting to run PROCESS ...")
        self._event_ready.wait()
        self._logger.debug("running PROCESS ...")

        try:
            while not self._event_terminate.is_set():
                self._event_play.wait()
                try:
                    for data in self._block_generator:
                        self._event_play.wait()
                        self._q.put(data.T, timeout=self._timeout)

                    # end of file reached
                    self._q.put(None, timeout=self._timeout)  # signal end of file
                    self._event_terminate.wait()  # keep alive until finished reading buffer
                    return

                except queue.Full:
                    if (
                        self._event_play.is_set()
                        and not self._event_terminate.is_set()
                        and config.IS_RUNNING.is_set()
                    ):
                        lvl = (
                            logging.ERROR if not config.IS_DEBUG_MODE else logging.DEBUG
                        )
                        self._logger.log(
                            lvl,
                            "JACK buffer is full. Check for an error in the callback.",
                        )
        except KeyboardInterrupt:
            self._logger.error("interrupted by user.")

    def terminate_members(self, msg=""):
        """
        Extending the functionality to `terminate_members()` the audio player specific
        components. This also stops the audio playback first with an optionally providing a log
        message.

        Parameters
        ----------
        msg : str, optional
            logging message
        """
        if self._event_play.is_set():
            self.stop(msg)

        try:
            self._sf.close()
        except AttributeError:
            pass
        try:
            self._block_generator.close()
        except AttributeError:
            pass
        try:
            self._q.close()
            self._q.join_thread()
        except AttributeError:
            pass

        super().terminate_members()

    @property
    def is_auto_play(self):
        """
        Returns
        -------
        bool
            if audio is supposed to be played after program start
        """
        return self._is_auto_play

    def play(self):
        """Run audio playback, which is realized only by setting the according event."""
        if not self._check_alive("run playback"):
            return

        if self._event_play.is_set():
            self._logger.info("playback was already running.")
            return

        self._logger.info(
            f'running file "{os.path.basename(self._file_name)}" playback.'
        )
        self._event_play.set()

    def stop(self, msg=""):
        """
        Pause audio playback, which is realized only by setting the according event.

        Parameters
        ----------
        msg : str, optional
            logging message
        """
        if not self._check_alive("stop playback"):
            return

        if not self._event_play.is_set():
            self._logger.info("playback was already stopped.")
            return

        if msg:
            self._logger.warning(msg)
        else:
            self._logger.info(
                f'stopping file "{os.path.basename(self._file_name)}" playback.'
            )

        # fill output ports with zeros
        for port in self._client.outports:
            port.get_array().fill(0)

        self._event_play.clear()

    def _process(self, _):
        """
        Process block of audio data. This implementation provides the output of read audio data
        from file. `None` in buffer is used to signal the end of the audio file. If that is
        reached the `JackPlayer` instance will be terminated. There is no internal functionality
        to restart the playback afterwards. That can only be achieved by starting a new
        `JackPlayer` instance.

        Returns
        -------
        numpy.ndarray
            generated block of audio data that will be delivered to JACK
        """
        if not self._event_play.is_set() or not config.IS_RUNNING.is_set():
            return None

        try:
            output_td = self._q.get_nowait()
            if output_td is None:
                self.terminate_members("finished file playback.")

            return output_td

        except (queue.Empty, OSError):
            if self._event_play.is_set():
                lvl = logging.ERROR if not config.IS_DEBUG_MODE else logging.DEBUG
                self._logger.log(
                    lvl, "JACK buffer is empty, maybe increase buffersize."
                )
