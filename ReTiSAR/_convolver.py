from copy import copy

import numpy as np
import pyfftw
from scipy import signal

from . import config, HeadTracker, tools
from ._filter_set import FilterSetMiro, FilterSetMultiChannel, FilterSetShConfig


class Convolver(object):
    """
    Basic class to provide convolution of an input signal by complex multiplication in the frequency domain. The filter
    length has to be smaller or identical to the system audio block size.

    Attributes
    ----------
    _filter : FilterSet
        FIR filter specification with loaded filter coefficients and block-wise pre-calculated in frequency domain
    _is_passthrough : bool
        if signals should be passed through by processing them with the `FilterSet` dirac attributes
    _rfft : pyfftw.FFTW
        FFTW library wrapper with a pre-calculated optimal scheme for fast real-time computation of the 1D real DFT
    _irfft : pyfftw.FFTW
        FFTW library wrapper with a pre-calculated optimal scheme for fast real-time computation of the 1D inverse
        real DFT
    """

    @staticmethod
    def create_instance_by_filter_set(filter_set, block_length=None, source_positions=None, shared_tracker_data=None):
        """
        Static method to instantiate a `Convolver` class or one if its deriving classes, depending on the given
        `FilterSet.Type`.

        Parameters
        ----------
        filter_set : FilterSet
            beforehand loaded FIR filter set with a specific `FilterSet.Type`
        block_length : int, optional
            system specific size of every audio block
        source_positions : list of int or list of float, optional
            rendered binaural source positions as list of tuple of azimuth (counterclockwise) and elevation in degrees,
            int or float values are possible
        shared_tracker_data : multiprocessing.Array, optional
            shared data array from an existing tracker instance for dynamic binaural rendering, see `HeadTracker`

        Returns
        -------
        Convolver, OverlapSaveConvolver, AdjustableFdConvolver or AdjustableShConvolver
            created instance according to `FilterSet.Type`
        """
        if not block_length:
            convolver = Convolver(filter_set)
        elif type(filter_set) == FilterSetMultiChannel\
                or (type(filter_set) == FilterSetMiro and filter_set.get_sh_configuration().arir_config):
            convolver = OverlapSaveConvolver(filter_set, block_length)
        elif type(filter_set) == FilterSetMiro and not filter_set.get_sh_configuration().arir_config:
            convolver = AdjustableShConvolver(filter_set, block_length, source_positions, shared_tracker_data)
        else:
            convolver = AdjustableFdConvolver(filter_set, block_length, source_positions, shared_tracker_data)

        # prevent running debugging help function in case of `AdjustableShConvolver` (needs to be invoked after
        # `AdjustableShConvolver.prepare_renderer_sh_processing()` instead!)
        if config.IS_DEBUG_MODE and type(convolver) is not AdjustableShConvolver:
            # noinspection PyTypeChecker
            convolver._debug_filter_block(len(source_positions) if source_positions else 0)

        return convolver

    def __init__(self, filter_set):
        """
        Initialize `Convolver` instance and trigger calculation of filter coefficients in frequency domain.

        Parameters
        ----------
        filter_set : FilterSet
            beforehand loaded FIR filter set
        """
        self._filter = filter_set
        self._is_passthrough = False
        self._rfft = None
        self._irfft = None

        # do not run if called by an inheriting class
        if type(self) is Convolver:
            self._filter.calculate_filter_blocks_fd(None)

    def __copy__(self):
        _filter = copy(self._filter)
        _filter.load(None, is_prevent_logging=True)
        new = type(self)(_filter)
        new.__dict__.update(self.__dict__)
        return new

    def __str__(self):
        return '[ID={}, _filter={}, _is_passthrough={}, _rfft=shape{}--shape{}, _irfft=shape{}--shape{}]'.format(
            id(self), self._filter, self._is_passthrough, self._rfft.input_shape, self._rfft.output_shape,
            self._irfft.input_shape, self._irfft.output_shape)

    def init_fft_optimize(self, logger=None):
        """
        Initialize `pyfftw` objects with given `config` parameters, for most efficient real-time DFT.

        Parameters
        ----------
        logger : logging.Logger, optional
            instance to provide identical logging behaviour as the parent process
        """
        log_str = 'initializing FFTW DFT optimization ...' if config.IS_PYFFTW_MODE else\
            'skipping FFTW DFT optimization.'
        logger.info(log_str) if logger else print(log_str)

        if not config.IS_PYFFTW_MODE:
            return

        self._rfft = pyfftw.builders.rfft(np.zeros_like(self._filter.get_dirac_td()), overwrite_input=True)
        self._irfft = pyfftw.builders.irfft(np.zeros_like(self._filter.get_dirac_blocks_fd()[0]), overwrite_input=False)

    def set_passthrough(self, new_state=None):
        """
        Parameters
        ----------
        new_state : bool or None, optional
            new passthrough state if processing should be virtually bypassed by using a dirac as a 'filter', if `None`
            is given this function works as a toggle between states

        Returns
        -------
        bool
            actually realized passthrough state
        """
        if new_state is None:
            self._is_passthrough = not self._is_passthrough
        else:
            self._is_passthrough = new_state
        return self._is_passthrough

    def _debug_filter_block(self, input_count, is_generate_noise=False):
        """
        Provides debugging possibilities the `filter_block()` function before running the `Convolver` as a separate
        process, where breakpoints do not work anymore. Arbitrary audio blocks can be send into the array,
        here some white noise in the shape to be expected from a `JackRenderer` input is generated.

        Parameters
        ----------
        input_count : int or None
            number of input channels expected
        is_generate_noise : bool, optional
            if white noise should be generated to test convolution, otherwise a dirac impulse will be used

        Returns
        -------
        numpy.ndarray
            filtered output blocks in time domain of size [number of output channels; `_block_length`]
        """
        # catch up with optimizing DFT, if had not been done yet
        if not self._rfft or not self._irfft:
            self.init_fft_optimize()

        # generate input blocks
        if not input_count:  # 0 or None
            input_count = self._rfft.input_shape[-2]

        # noinspection PyUnresolvedReferences
        block_length = self._rfft.input_shape[-1] if type(self) is Convolver else self._block_length
        if is_generate_noise:
            ip = tools.generate_noise((input_count, block_length))  # white noise
        else:
            ip = np.zeros((input_count, block_length))
            ip[:, 0] = 1  # dirac impulse

        # calculate filtered blocks
        op = self.filter_block(ip)

        # potentially check output blocks
        return op

    def filter_block(self, input_td):
        # transform into frequency domain
        input_fd = self._rfft(input_td) if config.IS_PYFFTW_MODE else np.fft.rfft(input_td)

        # do complex multiplication
        result_fd = input_fd * self._get_current_filters_fd()

        # transform back into time domain
        return self._irfft(result_fd) if config.IS_PYFFTW_MODE else np.fft.irfft(result_fd)

    def get_input_channel_count(self):
        """
        Returns
        -------
        int
            number of processed input channels
        """
        # noinspection PyProtectedMember
        return self._filter._irs_td.shape[0]

    def get_output_channel_count(self):
        """
        Returns
        -------
        int
            number of processed output channels
        """
        # noinspection PyProtectedMember
        return self._filter._irs_td.shape[0] * self._filter._irs_td.shape[1]

    def _get_current_filters_fd(self):
        """
        Returns
        -------
        numpy.ndarray
            complex one-sided filter frequency spectra to be applied to the signal (based on current passthrough state
            and position) of size [number of blocks; number of output channels; block length (+1 depending on even or
            uneven length)]
        """
        if self._is_passthrough:
            return self._filter.get_dirac_blocks_fd()

        return self._filter.get_filter_blocks_fd()


class OverlapSaveConvolver(Convolver):
    """
    Extension of `Convolver` to allow convolution of filter lengths independent of the system audio block size. This is
    achieved by processing the complex multiplication in the frequency domain based on the overlap-save-algorithm.

    Attributes
    ----------
    _block_length : int
        system wide time domain audio block size
    _blocks_fd : numpy.ndarray
        complex one-sided frequency spectra contained in a shifting buffer of size [number of blocks; number of
        output channels; `_block_length` (+1 depending on even or uneven length)]
    _input_block_td : numpy.ndarray
        time domain input samples contained in a shifting buffer of size [number of input channels; 2 * `_block_length`]
    _input_subsonic : tuple of numpy.ndarray or None
        subsonic highpass filter (b and a) coefficients applied to the input signals
    """

    def __init__(self, filter_set, block_length):
        """
        Extends the function of `Convolver` to initialize an overlap-add-convolver by also allocating the necessary
        buffer arrays for the block-wise processing.

        Parameters
        ----------
        block_length : int
            system wide time domain audio block size
        """
        super(OverlapSaveConvolver, self).__init__(filter_set)
        self._block_length = block_length
        self._input_block_td = None
        self._input_subsonic = None, None

        # calculate filter in frequency domain
        self._filter.calculate_filter_blocks_fd(self._block_length)
        self._blocks_fd = np.zeros_like(self._filter.get_dirac_blocks_fd())  # also inherit `dtype`

        # do not run if called by an inheriting class
        if type(self) is OverlapSaveConvolver:
            # reserve input block buffer according to input channel count
            self._input_block_td = np.zeros((self._blocks_fd.shape[-2], self._block_length * 2))

    def __copy__(self):
        _filter = copy(self._filter)
        _filter.load(self._block_length, is_prevent_logging=True)
        new = type(self)(_filter, self._block_length)
        new.__dict__.update(self.__dict__)
        return new

    def __str__(self):
        return '[{}, _block_length={}, _blocks_fd=shape{}, _input_block_td=shape{}]'.format(
            super(OverlapSaveConvolver, self).__str__()[1:-1], self._block_length, self._blocks_fd.shape,
            self._input_block_td.shape)

    def init_subsonic(self, cutoff_freq, fs, logger=None, is_iir=True):
        """
        Initialize subsonic input filter by calculating IIR or FIR filter coefficients.

        Parameters
        ----------
        cutoff_freq : float
            cutoff frequency of generated highpass filter
        fs : int
            sampling frequency of generated filter
        logger : logging.Logger, optional
            instance to provide identical logging behaviour as the parent process
        is_iir : bool, optional
            if IIR filter should be generated, otherwise a minimal phase FIR filter will be generated
        """
        if is_iir:
            log_str = 'pre-calculating subsonic IIR input filter components ...'
            logger.info(log_str) if logger else print(log_str)

            b, a = tools.generate_highpass_td(True, fs, cutoff_freq, iir_order=4)
            self._input_subsonic = b, a

            # generate plot
            _, b_a_fd = signal.freqz(b, a, int(self._block_length/2), whole=False, fs=fs)
            tools.export_plot(tools.plot_ir_and_tf(b_a_fd[np.newaxis, :], fs), 'subsonic', logger=logger)

        else:
            log_str = 'pre-calculating subsonic linear phase FIR input filter components ...'
            logger.info(log_str) if logger else print(log_str)

            b, _ = tools.generate_highpass_td(False, fs, cutoff_freq, fir_samples=self._block_length * 2)
            # shift by block length (see arrangement `_filter_block_shift_and_convert_input`
            b = np.roll(b, self._block_length)

            # rfft not replaced by `pyfftw` since it is not executed in real-time
            b_fd = np.fft.rfft(b)[np.newaxis, :]
            self._input_subsonic = b_fd, None

            # generate plot
            tools.export_plot(tools.plot_ir_and_tf(b_fd, fs), 'subsonic', logger=logger)

        # repeat running debugging help function with subsonic filter
        if config.IS_DEBUG_MODE:
            self._debug_filter_block(None)

    def init_fft_optimize(self, logger=None):
        """
        Initialize `pyfftw` objects with given `config` parameters, for most efficient real-time DFT.

        Parameters
        ----------
        logger : logging.Logger, optional
            instance to provide identical logging behaviour as the parent process
        """
        super(OverlapSaveConvolver, self).init_fft_optimize(logger)
        if not config.IS_PYFFTW_MODE:
            return

        self._rfft = pyfftw.builders.rfft(self._input_block_td, overwrite_input=True)
        self._irfft = pyfftw.builders.irfft(self._blocks_fd[0], overwrite_input=False)

    def filter_block(self, input_block_td):
        """
        Process a block of samples with the given `FilterSet`. Steps before the complex multiplication are provided by
        `_filter_block_shift_and_convert_input()`. Steps after the complex multiplication are provided by
        `_filter_block_shift_and_convert_result()`.

        `_blocks_fd[i]` will form the output in i blocks time, so each input block is multiplied by `filter_block_fd[i]`
        and summed into `_blocks_fd[i]`. After each block this queue is rotated to maintain this invariant.

        `_input_block_td` first half contains the input for this block and the second half contains the input from the
        previous one. That leads to half of each block in `_blocks_fd` will contain the tail from the previous one.

        In passthrough mode, the signal is still shifted in the buffer and transformed to frequency domain to deliver a
        smooth transition behaviour when toggling passthrough.

        Parameters
        ----------
        input_block_td : numpy.ndarray
            block of time domain input samples of size [number of input channels; `_block_length`]

        Returns
        -------
        numpy.ndarray
            block of filtered time domain output samples of size [number of output channels; `_block_length`]
        """
        # transform into frequency domain
        input_block_fd = self._filter_block_shift_and_convert_input(self._input_block_td, input_block_td, self._rfft,
                                                                    self._input_subsonic)

        if self._is_passthrough:
            # discard higher inputs than there are existing outputs
            input_block_fd = input_block_fd[:self._blocks_fd.shape[-2]]
            # just override the first buffer block
            self._blocks_fd[0, 0, :input_block_fd.shape[-2]] = input_block_fd
        else:
            # block-wise complex multiplication into buffer
            self._filter_block_complex_multiply(self._blocks_fd, self._get_current_filters_fd(), input_block_fd)

        # transform back into time domain
        output_block_td, self._blocks_fd = self._filter_block_shift_and_convert_result(self._blocks_fd, self._irfft)
        return output_block_td

    @staticmethod
    def _filter_block_shift_and_convert_input(buffer_block_td, input_block_td, rfft, subsonic_coefficients):
        """
        Parameters
        ----------
        buffer_block_td : numpy.ndarray
            reference to time domain input samples buffer of size [number of input channels; 2 * `_block_length`]
        input_block_td : numpy.ndarray
            block of time domain input samples of size [number of input channels; `_block_length`]
        rfft : pyfftw.FFTW or None
            instantiated FFTW library wrapper with a pre-calculated optimal scheme for fast real-time computation of
            the 1D real DFT

        Returns
        -------
        numpy.ndarray
            block of complex one-sided input frequency spectra of size [number of input channels; `_block_length` (+1
            depending on even or uneven length)]
        """
        # # set new input to beginning of stored blocks (after shifting forwards)
        # buffer_block_td[:, input_block_td.shape[1]:] = buffer_block_td[:, :input_block_td.shape[1]]
        # buffer_block_td[:, :input_block_td.shape[1]] = input_block_td

        # set new input to end of stored blocks (after shifting backwards)
        buffer_block_td[:, :input_block_td.shape[1]] = buffer_block_td[:, input_block_td.shape[1]:]
        buffer_block_td[:, input_block_td.shape[1]:] = input_block_td

        # TODO: check implementation, this should be done just on `input_block_td` beforehand?
        if subsonic_coefficients[1] is not None:
            # apply IIR subsonic
            buffer_block_td = signal.lfilter(subsonic_coefficients[0], subsonic_coefficients[1], buffer_block_td)

        # transform stored blocks into frequency domain
        buffer_block_fd = rfft(buffer_block_td) if config.IS_PYFFTW_MODE else np.fft.rfft(buffer_block_td)

        if subsonic_coefficients[1] is None and subsonic_coefficients[0] is not None:
            # apply FIR subsonic
            buffer_block_fd *= subsonic_coefficients[0]

        return buffer_block_fd

    @staticmethod
    def _filter_block_complex_multiply(buffer_blocks_fd, filter_blocks_fd, input_block_fd):
        """
        Parameters
        ----------
        buffer_blocks_fd : numpy.ndarray
            reference to complex one-sided frequency spectra of size [number of blocks; number of output channels;
            `_block_length` (+1 depending on even or uneven length)]
        filter_blocks_fd : numpy.ndarray
            complex one-sided filter frequency spectra to be applied to the signal of size [number of blocks; number
            of input channels; number of output channels; block length (+1 depending on even or uneven length)]
        input_block_fd : numpy.ndarray
            block of complex one-sided input frequency spectra of size [number of input channels; `_block_length` (+1
            depending on even or uneven length)]
        """
        # do block-wise complex multiplication
        for block_fd, filter_block_fd in zip(buffer_blocks_fd, filter_blocks_fd):
            block_fd += filter_block_fd * input_block_fd

    @staticmethod
    def _filter_block_shift_and_convert_result(buffer_blocks_fd, irfft):
        """
        Parameters
        ----------
        buffer_blocks_fd : numpy.ndarray
            reference to complex one-sided frequency spectra of size [number of blocks; number of output channels;
            `_block_length` (+1 depending on even or uneven length)]
        irfft : pyfftw.FFTW or None
            instantiated FFTW library wrapper with a pre-calculated optimal scheme for fast real-time computation of
            the 1D inverse real DFT

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            block of filtered time domain output samples of size [number of output channels; `_block_length`],
            reference to complex one-sided frequency spectra of size [number of blocks; number of output channels;
            `_block_length` (+1 depending on even or uneven length)]
        """
        # transform first block back into time domain
        first_block_td = irfft(buffer_blocks_fd[0]) if config.IS_PYFFTW_MODE else np.fft.irfft(buffer_blocks_fd[0])

        # shift blocks forwards
        if buffer_blocks_fd.shape[0] > 1:
            # since `buffer_blocks_fd` is assigned a new copy of an ndarray, it needs to be returned
            buffer_blocks_fd = np.roll(buffer_blocks_fd, -1, axis=0)

        # set last block to zero
        buffer_blocks_fd[-1] = 0.0

        # remove 1st singular dimension and return relevant first half of the time domain data
        # return first_block_td[0, :, :int(first_block_td.shape[-1] / 2)], buffer_blocks_fd
        # half of 1st block is not in C-order, but copy() did not have a performance impact

        # remove 1st singular dimension and return relevant second half of the time domain data
        return first_block_td[0, :, int(first_block_td.shape[-1] / 2):], buffer_blocks_fd
        # half of 1st block is not in C-order, but copy() did not have a performance impact


class AdjustableFdConvolver(OverlapSaveConvolver):
    """
    Extension of `OverlapSaveConvolver` to allow fast convolution in frequency domain with exchangeable impulse
    responses, i.e. for binaural rendering (selection of a Head Related Impulse Response depending on the head
    orientation).

    Attributes
    ----------
    _tracker_deg : multiprocessing.Array
        currently received head position received from a chosen kind of tracking device, see `HeadTracker.DataIndex`
    _sources_deg : numpy.ndarray
        rendered binaural source positions of size [number of sources; 2] with second dimension being azimuth
        (counterclockwise) and elevation in degrees
    _is_crossfade : bool
        if output signals applying the current filter should be cross-faded with the output signals of the last filter
        (theoretically improving the perceptual appeal when exchanging filters i.e. of an HRIR, almost doubles the
        processing effort)
    _window_out_td : numpy.ndarray
        time domain window samples to fade out during a block of size [`_block_length`]
    _window_in_td : numpy.ndarray
        time domain window samples to fade in during a block of size [`_block_length`]
    _last_filters_fd : numpy.ndarray
        complex one-sided filter frequency spectra that were applied to the signal in the last processing frame of
        size like `_get_current_filters_fd()`
    _last_blocks_fd : numpy.ndarray
        complex one-sided frequency spectra that had the past last processing filters applied to the signal contained
        in a shifting buffer of size like `_blocks_fd`
    """

    def __init__(self, filter_set, block_length, source_positions, shared_tracker_data):
        """
        Extends the function of `OverlapSaveConvolver` to initialize a binaural renderer by loading the provided
        positions of virtual sound sources.

        Parameters
        ----------
        source_positions : tuple of int or tuple of float
            rendered binaural source positions as list of tuple of azimuth (counterclockwise) and elevation in degrees,
            int or float values are possible
        shared_tracker_data : multiprocessing.Array
            shared data array from an existing tracker instance for dynamic binaural rendering, see `HeadTracker`

        Raises
        ------
        ValueError
            in case no `source_positions` are given
        """
        super(AdjustableFdConvolver, self).__init__(filter_set, block_length)
        self._is_crossfade = True

        if not source_positions:
            raise ValueError('no virtual source positions (according to playback file channel count) given.')
        self._tracker_deg = shared_tracker_data
        self._sources_deg = np.array(source_positions, dtype=np.float16)

        # limit to amount of output channels on second dimension to 1
        self._blocks_fd = self._blocks_fd[:, :1]

        # discard higher dimensions than there are existing inputs
        self._input_block_td = np.zeros((len(source_positions), self._block_length * 2))

        # calculate COSINE-Square cross-fade windows
        self._window_out_td = np.arange(self._block_length, dtype=self._input_block_td.dtype)
        self._window_out_td = np.square(np.cos(self._window_out_td * np.pi / (2 * (self._block_length - 1))))
        self._window_in_td = np.flip(self._window_out_td).copy()

        # allocate space for storing buffers relevant to cross-fading
        self._last_blocks_fd = np.zeros_like(self._blocks_fd)
        self._last_filters_fd = np.zeros_like(self._get_current_filters_fd())

    def __copy__(self):
        _filter = copy(self._filter)
        _filter.load(self._block_length, is_prevent_logging=True)
        new = type(self)(_filter, self._block_length, self._sources_deg, self._tracker_deg)
        new.__dict__.update(self.__dict__)
        return new

    def __str__(self):
        return '[{},  _tracker_deg=len({}), _sources_deg=shape{}, _is_crossfade={}, _window_td=shape{}]'.format(
            super(AdjustableFdConvolver, self).__str__()[1:-1], len(self._tracker_deg), self._sources_deg.shape,
            self._is_crossfade, self._window_out_td.shape)

    def filter_block(self, input_block_td):
        """
        Process a block of samples with the given `FilterSet`. Steps before the complex multiplication are provided by
        `_filter_block_shift_and_convert_input()`. Steps after the complex multiplication are provided by
        `_filter_block_shift_and_convert_result()`.

        In passthrough mode, the functionality of `OverlapSaveConvolver` filtering is used.

        Parameters
        ----------
        input_block_td : numpy.ndarray
            block of time domain input samples of size [number of input channels; `_block_length`]

        Returns
        -------
        numpy.ndarray
            block of filtered time domain output samples of size [number of output channels; `_block_length`]
        """
        if self._is_passthrough:
            return super(AdjustableFdConvolver, self).filter_block(input_block_td)

        # transform into frequency domain
        input_block_fd = self._filter_block_shift_and_convert_input(self._input_block_td, input_block_td, self._rfft,
                                                                    self._input_subsonic)
        filters_blocks_fd = self._get_current_filters_fd()

        # block-wise complex multiplication into current buffer
        self._filter_block_complex_multiply(self._blocks_fd, filters_blocks_fd, input_block_fd)
        # transform back into time domain
        output_in_block_td, self._blocks_fd = self._filter_block_shift_and_convert_result(self._blocks_fd, self._irfft)

        # skip further calculations in case no crossfade in time domain should be done
        if not self._is_crossfade:
            return output_in_block_td

        # block-wise complex multiplication into last buffer
        self._filter_block_complex_multiply(self._last_blocks_fd, self._last_filters_fd, input_block_fd)
        # transform back into time domain
        output_out_block_td, self._last_blocks_fd = self._filter_block_shift_and_convert_result(self._last_blocks_fd,
                                                                                                self._irfft)

        # store last used filters
        self._last_filters_fd = filters_blocks_fd  # copy is not necessary here (ndarray was newly created)

        # add in time domain after applying windows
        return (output_in_block_td * self._window_in_td) + (output_out_block_td * self._window_out_td)

    @staticmethod
    def _filter_block_complex_multiply(buffer_blocks_fd, filters_blocks_fd, input_block_fd):
        """
        Parameters
        ----------
        buffer_blocks_fd : numpy.ndarray
            reference to complex one-sided frequency spectra of size [number of blocks; number of output channels;
            `_block_length` (+1 depending on even or uneven length)]
        filters_blocks_fd : numpy.ndarray
            complex one-sided filter frequency spectra to be applied to the signal (based on current passthrough state
            and position) of size [number of sources, number of blocks; number of output channels; block length (+1
            depending on even or uneven length)]
        input_block_fd : numpy.ndarray
            block of complex one-sided input frequency spectra of size [number of input channels; `_block_length` (+1
            depending on even or uneven length)]
        """
        # summation for every source
        for s in range(filters_blocks_fd.shape[0]):
            # do block-wise complex multiplication
            for block_fd, filter_block_fd in zip(buffer_blocks_fd, filters_blocks_fd[s]):
                # division by `_sources_deg.shape[0]` is level adjustment in case multiple sources are rendered
                block_fd[0] += filter_block_fd * input_block_fd[s] / filters_blocks_fd.shape[0]

    def set_crossfade(self, new_state=None):
        """
        Parameters
        ----------
        new_state : bool or None, optional
            new crossfade state if output signals applying the current filter should be cross-faded with the output
            signals of the last filter, if `None` is given this function works as a toggle between states

        Returns
        -------
        bool
            actually realized crossfade state
        """
        if new_state is None:
            self._is_crossfade = not self._is_crossfade
        else:
            self._is_crossfade = new_state

        # clean buffers if crossfade was turned off
        if not self._is_crossfade:
            self._last_blocks_fd.fill(0)
            self._last_filters_fd.fill(0)

        return self._is_crossfade

    def get_input_channel_count(self):
        """
        Returns
        -------
        int
            number of rendered binaural source positions
        """
        return self._sources_deg.shape[0]

    def get_output_channel_count(self):
        """
        Returns
        -------
        int
            number of processed output channels
        """
        # noinspection PyProtectedMember
        return self._filter._irs_td.shape[1]

    def _get_current_filters_fd(self):
        """
        Returns
        -------
        numpy.ndarray
            complex one-sided filter frequency spectra to be applied to the signal (based on current passthrough state
            and position) of size [number of sources; number of blocks; number of output channels; block length (+1
            depending on even or uneven length)]
        """
        if self._is_passthrough:
            # return np.repeat(self._filter.get_dirac_blocks_fd(), self._sources_deg.shape[0], axis=1)[np.newaxis, :]
            return self._filter.get_dirac_blocks_fd()[np.newaxis, :]

        azims_deg, elevs_deg = self._calculate_individual_directions()
        # for s in range(azims_deg.shape[0]):
        #     print('source {} AZIM head {:>+6.1f} deg, source relative {:>3.0f} deg'.format(
        #         s, self._tracker_deg[HeadTracker.DataIndex.AZIM], azims_deg[s]))
        #     print('source {} ELEV head {:>+6.1f} deg, source relative {:>3.0f} deg'.format(
        #         s, self._tracker_deg[HeadTracker.DataIndex.ELEV], elevs_deg[s]))

        # stack for all sources
        return np.stack([self._filter.get_filter_blocks_fd(azim_deg, elev_deg)
                         for azim_deg, elev_deg in zip(azims_deg, elevs_deg)])

    def _calculate_individual_directions(self):
        """
        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            rotation azimuth angle in degrees (wrapped to be between 0 and 359) of size [number of sources],
            rotation elevation angle in degrees (wrapped to be between -180 and 179) of size [number of sources]
        """
        # invert tracker direction in case a BRIR (not HRIR) is rendered
        # noinspection PyProtectedMember
        tracker_dir = 1 if self._filter._is_hrir else -1

        azims_deg = tracker_dir * self._tracker_deg[HeadTracker.DataIndex.AZIM] + self._sources_deg[:, 0]
        azims_deg = (azims_deg + 360.0) % 360.0  # azimuth between 0 and 359

        elevs_deg = tracker_dir * self._tracker_deg[HeadTracker.DataIndex.ELEV] + self._sources_deg[:, 1]
        elevs_deg = ((elevs_deg + 180.0) % 360.0) - 180.0  # elevation between -180 and 179

        # use floats to calculate above to preserve `_sources_deg` dtype
        return azims_deg, elevs_deg


class AdjustableShConvolver(AdjustableFdConvolver):
    """
    Extension of `AdjustableFdConvolver` to allow fast convolution in spherical harmonics domain while adjusting
    orientation by a rotation matrix, i.e. for binaural rendering (depending on the head  orientation).

    Attributes
    ----------
    _sh_m : numpy.ndarray
        set of spherical harmonics orders of size [count according to `sh_max_order`]
    _sh_m_rev_id : list of int
        set of reversed spherical harmonics orders indices of size [count according to `sh_max_order`]
    _sh_bases_weighted : numpy.ndarray
        spherical harmonic bases weighted by grid weights of spatial sampling points of size [number according to
        `sh_max_order`; number of input channels]
    _last_sh_azim_nm : numpy.ndarray
        set of spherical harmonics azimuth weights that were applied to the signal in the last processing frame of
        size [count according to `sh_max_order`; 1; 1]
    """

    def __init__(self, filter_set, block_length, source_positions, shared_tracker_data):
        """
        Extends the function of `OverlapSaveConvolver` to initialize a binaural renderer by loading the provided
        positions of virtual sound sources.

        Parameters
        ----------
        source_positions : tuple of int or tuple of float
            rendered binaural source positions as list of tuple of azimuth (counterclockwise) and elevation in degrees,
            int or float values are possible
        shared_tracker_data : multiprocessing.Array
            shared data array from an existing tracker instance for dynamic binaural rendering, see `HeadTracker`

        Raises
        ------
        NotImplementedError
            in case more then one `source_positions` are given
        """
        super(AdjustableShConvolver, self).__init__(filter_set, block_length, source_positions, shared_tracker_data)

        if self._sources_deg.shape[0] > 1:
            raise NotImplementedError('more then one virtual source position given, which is not supported yet.')

        self._sh_m = None
        self._sh_m_rev_id = None
        self._sh_bases_weighted = None
        self._last_filters_fd = None
        self._last_sh_azim_nm = None

    def __copy__(self):
        # _filter = copy(self._filter)
        # _filter.load(self._block_length, is_prevent_logging=True)
        # new = type(self)(_filter, self._block_length, self._sources_deg, self._tracker_deg)
        # new.__dict__.update(self.__dict__)
        # # this is supposed to be the filter of the pre-renderer !!!
        # new.prepare_sh_processing(_filter.get_sh_configuration(), 0)
        # return new
        raise NotImplementedError('This implementation was not been tested so far.')

    def __str__(self):
        return '[{}, _sh_m=shape{}, _sh_m_rev_id=shape{}, _sh_bases_weighted=shape{}]'.format(
            super(AdjustableFdConvolver, self).__str__()[1:-1], self._sh_m.shape, self._sh_m_rev_id.shape,
            self._sh_bases_weighted.shape)

    def prepare_sh_processing(self, input_sh_config, amp_limit_db, logger=None):
        """
        Calculate components which can be prepared before spherical harmonic processing in real-time. This contains
        calculating all spherical harmonics orders, coefficients and base functions. Also a modal radial filter
        according to the provided input array configuration will be generated and applied preliminary.

        Parameters
        ----------
        input_sh_config : FilterSetShConfig
            combined filter configuration with all necessary information to transform an incoming audio block into
            spherical harmonics sound field coefficients in real-time
        amp_limit_db : int
            maximum modal amplification limit in dB
        logger : logging.Logger, optional
            instance to provide identical logging behaviour as the parent process
        """

        def _calc_reversed_m_ids(max_order):
            """
            Parameters
            ----------
            max_order : int
                maximum spherical harmonics order

            Returns
            -------
            list of int
                set of reversed spherical harmonics orders indices of size [number according to `max_order`]
            """
            m_ids = list(range(max_order * (max_order + 2) + 1))
            for o in range(max_order + 1):
                id_start = o ** 2
                id_end = id_start + o * 2 + 1
                m_ids[id_start:id_end] = reversed(m_ids[id_start:id_end])
            return m_ids

        # prepare attributes
        self._sh_m = input_sh_config.sh_m.copy()  # copy to ensure C-order
        # noinspection PyProtectedMember
        self._sh_m_rev_id = _calc_reversed_m_ids(self._filter._sh_max_order)
        self._sh_bases_weighted = input_sh_config.sh_bases_weighted.copy()  # copy to ensure C-order
        self._last_sh_azim_nm = np.zeros((self._sh_bases_weighted.shape[0], 1, 1), dtype=self._sh_bases_weighted.dtype)

        # prepare block buffers
        self._filter.calculate_filter_blocks_nm(logger=logger)
        self._filter.apply_radial_filter(input_sh_config.arir_config, amp_limit_db, logger=logger)

        # adjust buffer block sizes according to array configuration
        arir_channel_count = input_sh_config.sh_bases_weighted.shape[-1]
        self._input_block_td = np.zeros((arir_channel_count, self._block_length * 2))

        # catch up on running debugging help function in case of `AdjustableShConvolver`
        if config.IS_DEBUG_MODE:
            self._debug_filter_block(None)

    def get_input_channel_count(self):
        """
        Returns
        -------
        int
            number of processed input channels
        """
        return self._input_block_td.shape[-2]

    def filter_block(self, input_block_td):
        """
        Process a block of samples with the given `FilterSet`. Steps before the complex multiplication are provided by
        `_filter_block_shift_and_convert_input()`. Steps after the complex multiplication are provided by
        `_filter_block_shift_and_convert_result()`.

        In passthrough mode, the functionality of `OverlapSaveConvolver` filtering is used.

        Parameters
        ----------
        input_block_td : numpy.ndarray
            block of time domain input samples of size [number of input channels; `_block_length`]

        Returns
        -------
        numpy.ndarray
            block of filtered time domain output samples of size [number of output channels; `_block_length`]
        """

        def _spat_ft(block_fd, sh_bases_weighted):
            """
            Spatial Fourier transform optimized for periodic calculation in real time.

            Parameters
            ----------
            block_fd : numpy.ndarray
                block of one-sided complex frequency spectra of size [number of input channels; block length (+1
                depending on even or uneven length)]
            sh_bases_weighted : numpy.ndarray
                spherical harmonic bases weighted by grid weights of spatial sampling points of size [number according
                to `sh_max_order`; number of input channels]

            Returns
            -------
            numpy.ndarray
                block of complex spherical harmonics coefficients of size [number according to `sh_order`; block length
                (+1 depending on even or uneven length)]
            """
            return np.dot(sh_bases_weighted, block_fd)

        if self._is_passthrough:
            return super(AdjustableShConvolver, self).filter_block(input_block_td)

        # transform into frequency domain and sh-coefficients
        input_block_nm = _spat_ft(self._filter_block_shift_and_convert_input(self._input_block_td, input_block_td,
                                                                             self._rfft, self._input_subsonic),
                                  self._sh_bases_weighted)

        # # _TODO: implement block-wise processing if filter is longer
        # for filter_block_nm, block_nm in zip(self._filter.get_filter_blocks_nm(), self._blocks_nm):
        #     block_nm[0] += filter_block_nm * input_block_nm * ...

        # apply reverse index and adjust size according to filter channels
        input_block_nm = input_block_nm[self._sh_m_rev_id][:, np.newaxis, :]
        input_block_nm = np.repeat(input_block_nm, self._blocks_fd.shape[-2], axis=1)
        # apply HRIR coefficients
        input_block_nm *= self._filter.get_filter_blocks_nm()[0]

        # get head-tracker position (neglect elevation)
        azim_rad, _ = self._calculate_individual_directions()
        sh_azim_nm = np.exp(-1j * self._sh_m * azim_rad)[:, np.newaxis, np.newaxis]

        # calculation back into frequency domain into current buffer, after applying rotation coefficients
        self._blocks_fd[0, 0] = np.sum(input_block_nm * sh_azim_nm, axis=0)
        # transform back into time domain
        output_in_block_td, self._blocks_fd = self._filter_block_shift_and_convert_result(self._blocks_fd, self._irfft)

        # skip further calculations in case no crossfade in time domain should be done
        if not self._is_crossfade:
            return output_in_block_td

        # calculation back into frequency domain into last buffer, after applying rotation coefficients
        self._last_blocks_fd[0, 0] = np.sum(input_block_nm * self._last_sh_azim_nm, axis=0)
        # transform back into time domain
        output_out_block_td, self._last_blocks_fd = self._filter_block_shift_and_convert_result(self._last_blocks_fd,
                                                                                                self._irfft)

        # store last used azimuth exponents
        self._last_sh_azim_nm = sh_azim_nm  # copy should not be necessary here

        # add in time domain after applying windows
        return (output_in_block_td * self._window_in_td) + (output_out_block_td * self._window_out_td)

    def _calculate_individual_directions(self):
        """
        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            rotation azimuth angle in radians (was wrapped to be between 0 and 359) of size [number of sources],
            rotation elevation angle in radians (was wrapped to be between -180 and 179) of size [number of sources]
        """
        # noinspection PyProtectedMember
        azims_deg, elevs_deg = super(AdjustableShConvolver, self)._calculate_individual_directions()
        # inverse values since were not rotating virtual sources, but an actual sound field
        return np.deg2rad(-azims_deg), np.deg2rad(-elevs_deg)

    def set_crossfade(self, new_state=None):
        """
        Parameters
        ----------
        new_state : bool or None, optional
            new crossfade state if output signals applying the current filter should be cross-faded with the output
            signals of the last filter, if `None` is given this function works as a toggle between states

        Returns
        -------
        bool
            actually realized crossfade state
        """
        try:
            super(AdjustableShConvolver, self).set_crossfade(new_state)
        except AttributeError:
            pass

        # clean buffers if crossfade was turned off
        if not self._is_crossfade:
            self._last_sh_azim_nm.fill(0)

        return self._is_crossfade
