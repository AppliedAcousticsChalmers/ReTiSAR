from copy import copy

from . import Convolver, FilterSet, tools
from ._convolver import AdjustableFdConvolver, AdjustableShConvolver
from ._filter_set import FilterSetMiro, FilterSetShConfig, FilterSetSofa
from ._jack_client import JackClient


class JackRenderer(JackClient):
    """
    Extension of `JackClient` to provide functionality of processing audio signals in a block wise manner
    provided by the JACK client.

    Attributes
    ----------
    _is_passthrough : bool
        if JACK client should passthrough signals without any processing
    _convolver : Convolver
        providing block-wise processing of FIR filtering in the frequency domain
    """

    def __init__(self, name, block_length, filter_name=None, filter_type=None, source_positions=None,
                 shared_tracker_data=None, sh_max_order=None, sh_is_enforce_pinv=False, ir_trunc_db=None,
                 is_prevent_resampling=False, *args, **kwargs):
        """
        Extends the `JackClient` function to initialize a new JACK client and process. According to the
        documentation all attributes must be initialized in this function, to be available to the spawned process.

        Parameters
        ----------
        name : str
            name of the JACK client and spawned process
        block_length : int
            block length in samples of the JACK client (global setting for JACK server and all clients)
        filter_name : str or numpy.ndarray, optional
            file path/name of filter file or directly provided filter coefficients being used by `Convolver`
        filter_type : FilterSet.Type or string, optional
            type of filter being loaded
        source_positions : list of int or list of float, optional
            rendered binaural source positions as list of tuple of azimuth (counterclockwise) and elevation in degrees,
            int or float values are possible
        shared_tracker_data : multiprocessing.Array, optional
            shared data array from an existing tracker instance for dynamic binaural rendering, see `HeadTracker`
        sh_max_order : int, optional
            maximum spherical harmonics order used for the spatial Fourier transform
        sh_is_enforce_pinv : bool, optional
            if pseudo-inverse (Moore-Penrose) matrix will be used over explicitly given sampling grid weights to
            calculate the weighted SH basis functions, only relevant for filter sets in MIRO format
        ir_trunc_db : float
             impulse response truncation level in dB relative under peak
        is_prevent_resampling : bool, optional
            if loaded filter should not be resampled
        """
        super().__init__(name, block_length=block_length, *args, **kwargs)

        # set attributes
        self._is_passthrough = True
        self._convolver = None

        self._init_convolver(filter_name, filter_type, source_positions, shared_tracker_data, sh_max_order,
                             sh_is_enforce_pinv, ir_trunc_db, is_prevent_resampling)

    def _init_convolver(self, filter_name, filter_type, source_positions, shared_tracker_data, sh_max_order,
                        sh_is_enforce_pinv, ir_trunc_db, is_prevent_resampling):
        """
        Initialize `_convolver` specific attributes by also loading the necessary `FilterSet`. `filter_name` (and all
        other parameters can be `None`, so no `Convolver` is created.

        Parameters
        ----------
        filter_name : str or numpy.ndarray
            file path/name of filter file or directly provided filter coefficients being used by `Convolver`
        filter_type : FilterSet.Type or string
            type of filter being loaded
        source_positions : list of int or list of float
            rendered binaural source positions as list of tuple of azimuth (counterclockwise) and elevation in degrees,
            int or float values are possible
        shared_tracker_data : multiprocessing.Array
            shared data array from an existing tracker instance for dynamic binaural rendering, see `HeadTracker`
        sh_max_order : int
            maximum spherical harmonics order used for the spatial Fourier transform
        sh_is_enforce_pinv : bool, optional
            if pseudo-inverse (Moore-Penrose) matrix will be used over explicitly given sampling grid weights to
            calculate the weighted SH basis functions, only relevant for filter sets in MIRO format
        ir_trunc_db : float
             impulse response truncation level in dB relative under peak
        is_prevent_resampling : bool
            if loaded filter should not be resampled
        """
        filter_set = FilterSet.create_instance_by_type(file_name=filter_name, file_type=filter_type,
                                                       sh_max_order=sh_max_order, sh_is_enforce_pinv=sh_is_enforce_pinv)
        if filter_set is None:
            # no filter name or filter type given
            self._logger.warning('skipping filter load.')
            # use `_counter_dropout` as indicator if file was loaded
            self._counter_dropout = None
            return

        filter_set.load(block_length=self._client.blocksize, is_single_precision=self._is_single_precision,
                        logger=self._logger, ir_trunc_db=ir_trunc_db, check_fs=self._client.samplerate,
                        is_prevent_resampling=is_prevent_resampling, is_prevent_logging=self._logger.disabled)

        self._convolver = Convolver.create_instance_by_filter_set(filter_set=filter_set,
                                                                  block_length=self._client.blocksize,
                                                                  source_positions=source_positions,
                                                                  shared_tracker_data=shared_tracker_data)
        if type(self._convolver) == AdjustableFdConvolver and type(filter_set) in (FilterSetMiro, FilterSetSofa):
            self._logger.warning('selection of HRIR depending on head rotation is not properly implemented yet.')

        self._is_passthrough = False

    def client_register_and_connect_inputs(self, source_ports=True):
        """
        Register an identical number of input ports according to the provided source ports to the current client.
        Afterwards connect the given ports to the newly created target ports in a 1:1 relation.

        The number of input ports being registered is checked validated against the used `Convolver` instance.

        Parameters
        ----------
        source_ports : jack.Ports or bool or None, optional
            source ports for connecting the created input ports to. If None is given, a port number according to the
            system recording ports is registered. If True is given, ports will be registered and also connected to the
            system recording ports. If False is given, no ports will be registered or connected.

        Raises
        ------
        ValueError
            in case no source ports are provided and also no physical recording ports are found
        ValueError
            in case array processing will be used and the number of provided source ports is smaller than then array
            processing channels
        """
        if source_ports is False:
            return

        is_connect = source_ports is not None

        # temporarily pause execution
        event_ready_state_before = self._event_ready.is_set()
        self._event_ready.clear()

        if source_ports is None or source_ports is True:
            # get physical recording ports in case no ports were given
            source_ports = self._client.get_ports(is_physical=True, is_output=True)
            if not source_ports:
                raise ValueError('no source ports given and no physical recording ports detected.')

        # limit to source number specified by convolver
        convolver_port_count = self._convolver.get_input_channel_count()
        if len(source_ports) > convolver_port_count:
            self._logger.info(f'tried to connect {len(source_ports)} input ports ... limited the number to '
                              f'{convolver_port_count} (according to loaded filter).')
            source_ports = source_ports[:convolver_port_count]

        # elif len(source_ports) < convolver_port_count and type(self._convolver) is AdjustableShConvolver:
        #     raise ValueError(f'number of {len(source_ports)} input ports is smaller then the number of array '
        #                      f'processing channels {convolver_port_count} (according to loaded filter).')
        if len(source_ports) < convolver_port_count and type(self._convolver) is AdjustableShConvolver:
            self._logger.warning(f'skipping input register and connect.\n'
                                 f' --> number of {len(source_ports)} input ports is smaller then the number of array '
                                 f'processing channels {convolver_port_count} (according to loaded filter)')
        else:
            self._client_register_inputs(len(source_ports))

            if is_connect:
                # connect source to input ports
                for src, dst in zip(source_ports, self._client.inports):
                    self._client.connect(src, dst)

        # restore beforehand execution state
        if event_ready_state_before:
            self._event_ready.set()

    def _client_register_and_connect_outputs(self, target_ports=True):
        """
        Register a number of output ports according to the used `Convolver` instance to the current client in case none
        existed before. For further behaviour see documentation see called overridden function of `JackClient`.
        """
        if not self._convolver:
            return

        # register output ports if none exist
        if not len(self._client.outports):
            self._client_register_outputs(self._convolver.get_output_channel_count())

        # noinspection PyProtectedMember
        super()._client_register_and_connect_outputs(target_ports)

    def get_client_outputs(self):
        """
        Returns
        -------
        jack.Ports or list of int
            JACK client outports or list of convolver output channels, in case client was not started (yet)
        """
        if self.is_alive():
            return super().get_client_outputs()

        # in case client was not started yet, a list of output channels of the contained `Convolver` is returned
        self._logger.debug('output count determined from `Convolver`, since client was not started yet.')
        return list(range(self._convolver.get_output_channel_count()))

    def start(self, client_connect_target_ports=True):
        """
        Extends the `JackClient` function to `start()` the process. Here also the function concerning the JACK
        output ports suitable for binaural rendering is called.

        Parameters
        ----------
        client_connect_target_ports : jack.Ports or bool, optional
            see `_client_register_and_connect_outputs()` for documentation
        """
        super().start()

        # run after `AdjustableShConvolver.prepare_renderer_sh_processing()` was run
        self._convolver.init_fft_optimize(self._logger)

        self._logger.debug('activating JACK client ...')
        self._client.activate()
        self._client_register_and_connect_outputs(client_connect_target_ports)
        self._event_ready.set()

    def _process(self, input_td):
        """
        Process block of audio data. This implementation falls back to a straight passthrough behaviour if requested.
        Otherwise the provided `Convolver` instance will handle the signal processing.

        Parameters
        ----------
        input_td : numpy.ndarray
            block of audio data that was received from JACK

        Returns
        -------
        numpy.ndarray
            processed block of audio data that will be delivered to JACK
        """
        if self._is_passthrough:
            return super()._process(input_td)

        return self._convolver.filter_block(input_td)

    def set_client_passthrough(self, new_state=None):
        """
        Parameters
        ----------
        new_state : bool, int, float, str or None, optional
            new passthrough state of the client, if `None` is given this function works as a toggle between states

        Returns
        -------
        bool
            actually realized passthrough state
        """
        if not self._check_alive('set passthrough state'):
            return

        new_state = tools.transform_into_state(new_state, logger=self._logger)
        if not self._convolver and new_state is False:
            self._logger.warning('This client supports only passthrough mode.')
        else:
            # # let JackRenderer handle the passthrough behaviour
            # if new_state is None:
            #     self._is_passthrough = not self._is_passthrough
            # else:
            #     self._is_passthrough = new_state
            # self._logger.info(f'set passthrough state to {["OFF", "ON"][self._is_passthrough]}.')

            # let _convolver handle the passthrough behaviour
            new_state = self._convolver.set_passthrough(new_state)
            self._logger.info(f'set passthrough state to {["OFF", "ON"][new_state]}.')
            return new_state

    def set_client_crossfade(self, new_state=None):
        """
        Parameters
        ----------
        new_state : bool, int, float, str or None, optional
            new crossfade state of the client, if `None` is given this function works as a toggle between states

        Returns
        -------
        bool
            actually realized crossfade state
        """
        if not self._check_alive('set crossfade state'):
            return

        new_state = tools.transform_into_state(new_state, logger=self._logger)
        if (new_state is True and
                (not self._convolver or type(self._convolver) not in [AdjustableFdConvolver, AdjustableShConvolver])):
            self._logger.warning('This client does not support crossfade mode.')
        else:
            new_state = self._convolver.set_crossfade(new_state)
            self._logger.info(f'set crossfade state to {["OFF", "ON"][new_state]}.')
            return new_state

    def set_renderer_sh_order(self, new_order=None):
        """
        Parameters
        ----------
        new_order : int or float, optional
            new SH processing order of the renderer

        Returns
        -------
        bool
            actually realized SH processing order
        """
        if not self._check_alive('set SH processing order'):
            return
        if not type(self._convolver) is AdjustableShConvolver:
            self._logger.error(f'client is not rendering in spherical harmonics mode, "set SH processing order" '
                               f'ignored.')
            return
        if new_order is not None and int(new_order) != new_order:
            self._logger.error(f'invalid integer value "{new_order}", "set SH processing order" ignored.')
            return

        new_order = self._convolver.update_sh_processing(sh_new_order=new_order, logger=self._logger)
        self._logger.info(f'set SH processing order to {new_order:d}.')
        return new_order

    def prepare_renderer_sh_processing(self, input_sh_config, mrf_limit_db, compensation_type):
        """
        Calculate components which can be prepared before spherical harmonic processing in real-time. This contains
        calculating all spherical harmonics orders, coefficients and base functions. Also a modal radial filter
        according to the provided input array configuration will be generated and applied preliminary.

        Parameters
        ----------
        input_sh_config : FilterSetShConfig
            combined filter configuration with all necessary information to transform an incoming audio block into
            spherical harmonics sound field coefficients in real-time
        mrf_limit_db : int
            maximum modal amplification limit in dB
        compensation_type : str or Compensation.Type
            type of spherical harmonics processing compensation technique

        Raises
        ------
        ValueError
            in case the loaded `FilterSet` and `Convolver` instances are incompatible to spherical harmonics processing
        """
        self._logger.info('applying spherical harmonics configuration and pre-calculating components ...')

        if type(self._convolver) is not AdjustableShConvolver:
            # noinspection PyProtectedMember
            raise ValueError(f'convolver type {type(self._convolver)} of filter {type(self._convolver._filter)} is '
                             f'incompatible for spherical harmonics processing.')

        self._convolver.prepare_sh_processing(input_sh_config, mrf_limit_db, compensation_type, self._logger)

    def get_pre_renderer_sh_config(self):
        """
        Returns
        -------
        FilterSetShConfig
            combined filter configuration with all necessary information to transform an incoming audio block into
            spherical harmonics sound field coefficients in real-time
        """
        self._logger.info('gathering spherical harmonics configuration ...')

        # noinspection PyProtectedMember
        return self._convolver._filter.get_sh_configuration()


class JackRendererBenchmark(JackRenderer):
    """
    Extension of `JackRenderer` to provide functionality of processing multiple convolvers at the same time. This is
    meant to be used only for benchmarking purposes.

    Attributes
    ----------
    _convolvers : list of Convolver
        each instance providing block-wise processing of FIR filtering in the frequency domain
    """

    def __init__(self, name, block_length, filter_length, *args, **kwargs):
        """Extends the `JackRenderer` function to also initialize the list of contained `Convolver`."""

        def generate_dirac(length):
            import numpy as np

            # emulate impulse response by generating noise with the desired
            dtype = np.float32 if kwargs['is_single_precision'] else np.float64
            filter_ir = np.zeros(shape=(2, length), dtype=dtype)
            filter_ir[:, 0] = 1
            return filter_ir

        super().__init__(name, block_length, filter_name=generate_dirac(filter_length), filter_type='FIR_MULTICHANNEL',
                         *args, **kwargs)
        self._convolvers = [self._convolver]

    def add_convolver(self):
        """
        Add an additional `Convolver` instance by creating a clone of the initial one (identical but independent
        instances of all attributes).
        """
        self._convolvers.append(copy(self._convolver))

    def _process(self, input_td):
        """
        Process block of audio data. This implementation processes additional convolvers which output is discarded,
        hence this functionality is only useful for benchmarking purposes.

        Parameters
        ----------
        input_td : numpy.ndarray
            block of audio data that was received from JACK

        Returns
        -------
        numpy.ndarray
            processed block of audio data that will be delivered to JACK
        """
        # process main convolver
        output_td = super()._process(input_td)

        if len(self._convolvers) > 1:
            # process additional convolvers
            for c in self._convolvers[1:]:
                _ = c.filter_block(input_td)  # discard results

        return output_td
