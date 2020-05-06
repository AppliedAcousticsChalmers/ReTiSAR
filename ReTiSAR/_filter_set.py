import os
import sys
from collections import namedtuple
from enum import auto, Enum

import numpy as np
import samplerate
import sound_field_analysis as sfa
import soundfile

from . import tools


class FilterSet(object):
    """
    Flexible structure used to store FIR filter sets with an arbitrary number of channels. A given `FilterSet.Type`
    defines how individual input files are read and impulse responses are provided to a `Convolver` instance.

    All relevant time or frequency information of the contained filters are saved in multi-dimensional
    `numpy.ndarray`s for efficiency.

    Attributes
    ----------
    _file_name : str
        file path/name of filter source file
    _is_hrir : bool
        if filter set contains HRIR data
    _points_azim_deg : numpy.ndarray
        azimuth angles in degrees of dedicated head rotations at which the set contains impulse responses
    _points_elev_deg : numpy.ndarray
        elevation angles in degrees of dedicated head rotations at which the set contains impulse responses
    _fs : int
        filter data sampling rate
    _irs_td : numpy.ndarray
        filter impulse response in time domain of size [number of hrir positions; number of output channels; number
        of samples]
    _irs_blocks_fd : numpy.ndarray
        block-wise one-sided complex frequency spectra of the filters of size [number of blocks; number of hrir
        positions; number of output channels; block length (+1 depending on even or uneven length)]
    _dirac_td : numpy.ndarray
        dirac impulse in time domain of identical size like `_irs_td`
    _dirac_blocks_fd : numpy.ndarray
        block-wise one-sided complex frequency spectra of dirac impulses of identical size like `_irs_blocks_fd`
    """

    class Type(Enum):
        """
        Enumeration data type used to get an identification of loaded digital audio filter sets. It's attributes (with
        an arbitrary distinct integer value) are used as system wide unique constant identifiers.
        """

        FIR_MULTICHANNEL = auto()
        """
        Arbitrary number of channels used to convolve signals from input to output in a 1:1 relation.

        Used by `OverlapSaveConvolver` to equalize the binaural audio scene for headphones with size [fir_samples; 2].
        """

        HRIR_SSR = auto()
        """
        Head Related Impulse Responses in a shape as being shipped with the SoundScapeRenderer.
        https://ssr.readthedocs.io/en/latest/renderers.html#the-hrir-sets-shipped-with-ssr

        Used by `AdjustableFdConvolver` with size [fir_samples; 720].
        """

        BRIR_SSR = auto()
        """
        Binaural Room Impulse Responses in a shape as being shipped with the SoundScapeRenderer.
        https://ssr.readthedocs.io/en/latest/renderers.html#the-hrir-sets-shipped-with-ssr

        Used by `AdjustableFdConvolver` with size [fir_samples; 720].
        """

        HRIR_MIRO = auto()
        """
        Head Related Impulse Responses in a shape as being specified in the Measured Impulse Response Object.
        http://audiogroup.web.th-koeln.de/FILES/miro_documentation.pdf
        
        Used by `AdjustableFdConvolver` with flexible size specified by the measurement point grid.
        """

        ARIR_MIRO = auto()
        """
        Array Room Impulse Responses in a shape as being specified in the Measured Impulse Response Object.
        http://audiogroup.web.th-koeln.de/FILES/miro_documentation.pdf

        Used by `AdjustableFdConvolver` with flexible size specified by the measurement point grid.
        """

        AS_MIRO = auto()
        """
        Array audio stream with a configuration file as being specified in the Measured Impulse Response Object.
        http://audiogroup.web.th-koeln.de/FILES/miro_documentation.pdf

        Used by `AdjustableFdConvolver` with flexible size specified by the measurement point grid.
        """

        HRIR_SOFA = auto()
        """
        Head Related Impulse Responses in a shape as being specified in the Spatially Oriented Format for Acoustics.
        https://www.sofaconventions.org/mediawiki/index.php/SOFA_(Spatially_Oriented_Format_for_Acoustics)

        Currently not used or implemented. There doesn't seem to be an existing SOFA implementation for Python so far.
        """

        ARIR_SOFA = auto()
        """
        Array Room Impulse Responses in a shape as being specified in the Spatially Oriented Format for Acoustics.
        https://www.sofaconventions.org/mediawiki/index.php/SOFA_(Spatially_Oriented_Format_for_Acoustics)

        Currently not used or implemented. There doesn't seem to be an existing SOFA implementation for Python so far.
        """

    @staticmethod
    def create_instance_by_type(file_name, file_type, sh_order=None):
        """
        Parameters
        ----------
        file_name : str
            file path/name of filter source file
        file_type : str or FilterSet.Type
            type of filter source file
        sh_order : int, optional
            spherical harmonics order used for the spatial Fourier transform

        Returns
        -------
        FilterSetMultiChannel, FilterSetSsr, FilterSetMiro or FilterSetSofa
            created instance according to `FilterSet.Type`
        """
        _type = tools.transform_into_type(file_type, FilterSet.Type)
        if _type == FilterSet.Type.FIR_MULTICHANNEL:
            return FilterSetMultiChannel(file_name, is_hrir=False)
        elif _type == FilterSet.Type.HRIR_SSR:
            return FilterSetSsr(file_name, is_hrir=True)
        elif _type == FilterSet.Type.BRIR_SSR:
            return FilterSetSsr(file_name, is_hrir=False)
        elif _type == FilterSet.Type.HRIR_MIRO:
            return FilterSetMiro(file_name, is_hrir=True, sh_max_order=sh_order)
        elif _type in [FilterSet.Type.ARIR_MIRO, FilterSet.Type.AS_MIRO]:
            return FilterSetMiro(file_name, is_hrir=False, sh_max_order=sh_order)
        elif _type == FilterSet.Type.HRIR_SOFA:
            return FilterSetSofa(file_name, is_hrir=True)
        elif _type == FilterSet.Type.ARIR_SOFA:
            return FilterSetSofa(file_name, is_hrir=False)

    def __init__(self, file_name, is_hrir):
        """
        Initialize FIR filter, call `load()` afterwards to load the file contents!

        Parameters
        ----------
        file_name : str
            file path/name of filter source file
        is_hrir : bool
            if filter set contains HRIR data
        """
        self._file_name = file_name
        self._is_hrir = is_hrir

        self._points_azim_deg = None
        self._points_elev_deg = None
        self._fs = None
        self._irs_td = None
        self._irs_blocks_fd = None
        self._dirac_td = None
        self._dirac_blocks_fd = None

    def __str__(self):
        try:
            grid = '_points_azim_deg=shape{}, _points_elev_deg=shape{}, '.format(self._points_azim_deg.shape,
                                                                                 self._points_elev_deg.shape)
        except AttributeError:
            grid = ''
        return '[ID={}, _file_name={}, {}_fs={}, _irs_td=shape{}, _irs_blocks_fd=shape{}]'.format(
            id(self), os.path.relpath(self._file_name), grid, self._fs, self._irs_td.shape, self._irs_blocks_fd.shape)

    def load(self, block_length, logger=None, check_fs=None, is_prevent_resampling=False, is_prevent_logging=False,
             is_normalize=False, is_normalize_individually=False):
        """
        Loading the file contents of the provided FIR filter according to its specification. The loading procedures are
        are provided by the individual subclasses. This function also generates `_dirac_td` with identical size as
        `_irs_td`.

        Parameters
        ----------
        block_length : int or None
            system specific size of every audio block
        logger : logging.Logger, optional
            instance to provide identical logging behaviour as the parent process
        check_fs : int, optional
            global system sampling frequency, which is checked to be identical with the loaded filter
        is_prevent_logging : bool, optional
            prevent logging messages during load process
        is_prevent_resampling : bool, optional
            if loaded filter impulse response should not be resampled
        is_normalize : bool, optional
            if loaded filter impulse response should be normalized regarding its global peak
        is_normalize_individually : bool, optional
            if each channel of the loaded filter impulse response should be normalized regarding its individual peak
            (except for second to last dimension, resembling binaural pairs of left and right ear)
        """
        if not is_prevent_logging:
            self._log_load(logger)

        # type specific loading
        self._load()

        # check sampling rate match or initialize resampling
        if check_fs and check_fs != self._fs:
            self._resample(logger, check_fs, is_prevent_resampling)

        # normalize if necessary
        if is_normalize_individually or is_normalize:
            self._normalize(logger, is_individually=is_normalize_individually)

        # zero-padding if necessary
        self._zero_pad(block_length)

        # generate dirac impulse
        self._dirac_td = np.zeros(self._irs_td.shape[-2:])
        self._dirac_td[:, 0] = 1.0

    def _log_load(self, logger=None):
        """
        Log convenient status information about the audio file being read. If no `logger` is provided the information
        will just be output via regular `print()`.

        Parameters
        ----------
        logger : logging.Logger, optional
            instance to provide identical logging behaviour as the parent process
        """
        log_str = 'opening file "{}"'.format(os.path.relpath(self._file_name))

        # cut off name and reformat LF
        file_info = soundfile.info(self._file_name).__str__().split('\n', 1)[1].replace('\n', ', ')
        log_str += '\n --> {}'.format(file_info)

        logger.info(log_str) if logger else print(log_str)

    def _load(self):
        """Individual implementation of loading the FIR filter according to its specifications."""
        raise NotImplementedError('chosen filter type "{}" is not implemented yet.'.format(type(self)))

    def _resample(self, logger, target_fs, is_prevent_resampling=False):
        """
        Resample loaded filter to the given target sampling rate, utilizing the API of package `samplerate` to the
        C-library `libsamplerate`.

        Parameters
        ----------
        logger : logging.Logger
            instance to provide identical logging behaviour as the parent process
        target_fs : int
            target (system) sampling frequency
        is_prevent_resampling : bool, optional
            if loaded filter impulse response should not be resampled

        Raises
        ------
        ValueError
            in case resampling was prevented but would be necessary
        """
        _RESAMPLE_CONVERTER_TYPE = 'sinc_best'
        """Resampling converter type, see `samplerate` library."""

        log_str = 'resampling data from "{}"\n --> source: {} Hz, target: {} Hz'.format(
            os.path.relpath(self._file_name), self._fs, target_fs)
        if is_prevent_resampling:
            raise ValueError('prevented ' + log_str)
        logger.warning(log_str) if logger else print('[WARNING]  ' + log_str, file=sys.stderr)

        # initialize target variables
        ratio_fs = self._fs / target_fs

        if self._irs_td.ndim == 3:
            # iteration over first dimension, since `samplerate.resample()` only takes 2-dimensional inputs
            irs_td_new = np.stack([samplerate.resample(irs_td.T, ratio_fs, _RESAMPLE_CONVERTER_TYPE)
                                   for irs_td in self._irs_td])
            # restore former dimension order
            irs_td_new = np.swapaxes(irs_td_new, 1, 2).copy()  # copy to ensure C-order
        else:
            raise NotImplementedError('resampling for {} ndarray dimensions not implemented yet.'.format(
                self._irs_td.ndim))

        # update attributes
        self._irs_td = irs_td_new
        self._fs = target_fs

    def _normalize(self, logger, is_individually=False, target_amp=1.0):
        """
        Normalize loaded filter in time domain `_irs_td` to a maximum peak of 1.

        Parameters
        ----------
        logger : logging.Logger
            instance to provide identical logging behaviour as the parent process
        is_individually : bool, optional
            if all channels should be normalized independent of each other (except for second to last dimension,
            resembling binaural pairs of left and right ear)
        target_amp : float, optional
            target amplitude that should be scaled to
        """
        if is_individually:
            irs_peak = np.abs(self._irs_td).max(axis=-1).max(axis=-1)[:, np.newaxis, np.newaxis]
            log_str = '[INFO]  normalized {:d} IR peaks channel independent to {:.2f}.'.format(len(irs_peak), target_amp)
            logger.warning(log_str) if logger else print(log_str, file=sys.stderr)
        else:
            irs_peak = np.abs(self._irs_td).max()
            log_str = 'normalized IR peak from {:.2f} to {:.2f}.'.format(irs_peak, target_amp)
            logger.info(log_str) if logger else print(log_str)

        self._irs_td *= target_amp / irs_peak

    def _zero_pad(self, block_length):
        """
        Append zeros in time domain to `_irs_td` to full blocks and if filter length is smaller then provided block
        length.

        Parameters
        ----------
        block_length : int or None
            system wide length of audio blocks in samples
        """
        if not block_length:
            return
        block_count = self._irs_td.shape[2] / block_length
        if block_count == int(block_count) and self._irs_td.shape[-1] >= block_length:
            return

        # round up to multiple of block length
        block_count = np.math.ceil(block_count)

        # create padded array with otherwise identical dimensions
        irs_td_padded = np.zeros((self._irs_td.shape[0], self._irs_td.shape[1], block_length * block_count))
        irs_td_padded[:, :, :self._irs_td.shape[2]] = self._irs_td
        self._irs_td = irs_td_padded

    def calculate_filter_blocks_fd(self, block_length):
        """
        Split up the time domain information of the filter into blocks according to the provided length. This operation
        will add an additional dimension to the beginning of the block-wise `numpy.ndarray`s. For convenience in later
        processing the blocks will also be independently transformed into frequency domain and saved as complex
        one-sided spectra.

        The transformation process is also emulated for the dirac impulse response data. This is saved into
        `_dirac_blocks_fd` with identical size as `_irs_blocks_fd`.

        Parameters
        ----------
        block_length : int or None
            system wide length of audio blocks in samples
        """
        # entire signal is one block, if no size is given
        if not block_length:  # None or <=0
            block_length = self._irs_td.shape[-1]

        # for filter
        block_count = int(self._irs_td.shape[-1] / block_length)
        block_length_2 = block_length * 2
        # cut signal into slices and stack on new axis after transformation in frequency domain
        self._irs_blocks_fd = np.stack([np.fft.rfft(block_td, block_length_2)
                                        for block_td in np.dsplit(self._irs_td, block_count)])
        # rfft not replaced by `pyfftw` since it is not executed in real-time

        # for dirac impulse
        self._dirac_blocks_fd = np.zeros_like(self._irs_blocks_fd)[:, :1]  # limit to be rotation independent
        self._dirac_blocks_fd[0] = 1.0

    def get_dirac_td(self):
        """
        Returns
        -------
        numpy.ndarray
            dirac impulse in time domain of identical size like `get_filter_td()`
        """
        return self._dirac_td

    def get_dirac_blocks_fd(self):
        """
        Returns
        -------
        numpy.ndarray
            block-wise one-sided complex frequency spectra of dirac impulses of identical size like
            `get_filter_blocks_fd()`

        Raises
        ------
        RuntimeError
            in case requested blocks have not been calculated yet
        """
        if self._dirac_blocks_fd is None:
            raise RuntimeError('dirac blocks in frequency domain have not been calculated yet.')
        return self._dirac_blocks_fd

    def get_filter_td(self, azim_deg=0.0, elev_deg=0.0):
        """
        Parameters
        ----------
        azim_deg : int or float, optional
            azimuth of desired sound incidence direction in degrees (only relevant for HRIR)
        elev_deg: int or float, optional
            elevation of desired sound incidence direction in degrees (only relevant for HRIR)

        Returns
        -------
        numpy.ndarray
            filter impulse response in time domain of size [number of output channels; number of samples]
        """
        index = self._get_index_from_rotation(azim_deg, elev_deg)
        return self._irs_td[index]

    def get_filter_blocks_fd(self, azim_deg=0.0, elev_deg=0.0):
        """
        Parameters
        ----------
        azim_deg : int or float, optional
            azimuth of desired sound incidence direction in degrees (only relevant for HRIR)
        elev_deg: int or float, optional
            elevation of desired sound incidence direction in degrees (only relevant for HRIR)

        Returns
        -------
        numpy.ndarray
            block-wise one-sided complex frequency spectra of the filters of size [number of blocks; number of output
            channels; block length (+1 depending on even or uneven length)]

        Raises
        ------
        RuntimeError
            in case requested blocks have not been calculated yet
        """
        if self._irs_blocks_fd is None:
            raise RuntimeError('filter blocks in frequency domain have not been calculated yet.')

        index = self._get_index_from_rotation(azim_deg, elev_deg)
        # print('azim {:>-4.0f}, elev {:>-4.0f} -> index {:>4.0f}'.format(azim_deg, elev_deg, index))
        return self._irs_blocks_fd[:, index]

    def _get_index_from_rotation(self, azim_deg, elev_deg):
        """
        Parameters
        ----------
        azim_deg : int or float
            azimuth of desired sound incidence direction in degrees
        elev_deg: int or float
            elevation of desired sound incidence direction in degrees

        Returns
        -------
        int
            index in the `numpy.ndarray` storing the impulse response according to the desired incidence direction
        """
        raise NotImplementedError('chosen filter type "{}" is not implemented yet.'.format(type(self)))


class FilterSetMultiChannel(FilterSet):
    """
    Flexible structure used to store FIR filter sets with an arbitrary number of channels.

    All relevant time or frequency information of the contained filters are saved in multi-dimensional
    `numpy.ndarray`s for efficiency.
    """

    def _load(self):
        """
        Load an arbitrary number of impulse responses from multi-channel audio file. There is no directional information
        associated with this data. This can be used to provide a multi-channel-wise convolution, i.e. to equalize
        reproduction setups (headphones or loudspeakers).
        """
        self._irs_td, self._fs = soundfile.read(self._file_name, dtype=np.float32)
        self._irs_td = self._irs_td.T

        # append third dimension if not existing
        if self._irs_td.ndim < 3:
            self._irs_td = self._irs_td[np.newaxis, :]

        self._points_azim_deg = np.zeros(1, dtype=np.int16)
        self._points_elev_deg = np.zeros_like(self._points_azim_deg)

    def get_filter_blocks_fd(self, azim_deg=0.0, elev_deg=0.0):
        """
        Parameters
        ----------
        azim_deg : int or float, optional
            azimuth of desired sound incidence direction in degrees (only relevant for HRIR)
        elev_deg: int or float, optional
            elevation of desired sound incidence direction in degrees (only relevant for HRIR)

        Returns
        -------
        numpy.ndarray
            block-wise one-sided complex frequency spectra of the filters of size [number of blocks; 1; number of
            output channels; block length (+1 depending on even or uneven length)]

        Raises
        ------
        RuntimeError
            in case requested blocks have not been calculated yet
        """
        if self._irs_blocks_fd is None:
            raise RuntimeError('filter blocks in frequency domain have not been calculated yet.')

        return self._irs_blocks_fd

    def _get_index_from_rotation(self, azim_deg, elev_deg):
        """
        Parameters
        ----------
        azim_deg : int or float
            azimuth of desired sound incidence direction in degrees
        elev_deg: int or float
            elevation of desired sound incidence direction in degrees

        Returns
        -------
        int
            index in the `numpy.ndarray` storing the impulse response according to the desired incidence direction
        """
        return 0


class FilterSetSsr(FilterSet):
    """
    Flexible structure used to store Head Related Impulse Responses in a shape as being shipped with the
    SoundScapeRenderer.

    All relevant time or frequency information of the contained filters are saved in multi-dimensional
    `numpy.ndarray`s for efficiency.
    """

    def _load(self):
        """
        Load a filter set containing Head-Related Impulse Responses of 360 sources in equal distance on the horizontal
        plane. The file contains 720 audio channels, since the according IRs for each ear are provided consecutively for
        each source position.
        """
        irs_td, self._fs = soundfile.read(self._file_name, dtype=np.float32)
        irs_td_l = irs_td[:, 0:-1:2].T  # all even elements
        irs_td_r = irs_td[:, 1::2].T  # all uneven elements
        self._irs_td = np.swapaxes(np.stack((irs_td_l, irs_td_r)), 0, 1).copy()  # copy to ensure C-order

        self._points_azim_deg = np.linspace(0, 360, self._irs_td.shape[0], endpoint=False, dtype=np.int16)
        self._points_elev_deg = np.zeros_like(self._points_azim_deg)

    def _get_index_from_rotation(self, azim_deg, elev_deg):
        """
        Parameters
        ----------
        azim_deg : int or float
            azimuth of desired sound incidence direction in degrees
        elev_deg: int or float
            elevation of desired sound incidence direction in degrees

        Returns
        -------
        int
            index in the `numpy.ndarray` storing the impulse response according to the desired incidence direction
        """
        return int(np.floor(azim_deg))


class FilterSetMiro(FilterSet):
    """
    Flexible structure used to store Head Related Impulse Responses or Array Room Impulse Responses in a shape as being
    specified in the Measured Impulse Response Object.

    All relevant time, frequency or spherical harmonics information of the contained filters are saved in
    multi-dimensional `numpy.ndarray`s for efficiency.

    Attributes
    ----------
    _sh_max_order : int
        maximum spherical harmonics order used for the spatial Fourier transform
    _irs_grid : sfa.io.SphericalGrid
        recording / measurement point configuration
    _irs_blocks_nm : numpy.ndarray
        block-wise complex spherical harmonics coefficients of size [number of blocks; number according to `sh_order`;
        number of output channels; block length (+1 depending on even or uneven length)]
    _arir_config : sfa.io.ArrayConfiguration
        recording / measurement microphone array configuration
    """

    def __init__(self, file_name, is_hrir, sh_max_order=None):
        """
        Initialize FIR filter, call `load()` afterwards to load the file contents!

        Parameters
        ----------
        file_name : str
            file path/name of filter source file
        is_hrir : bool
            if filter set contains HRIR data
        sh_max_order : int, optional
            spherical harmonics order used for the spatial Fourier transform
        """
        super(FilterSetMiro, self).__init__(file_name, is_hrir)
        self._sh_max_order = sh_max_order

        self._irs_grid = None
        self._irs_blocks_nm = None
        self._arir_config = None

    def __str__(self):
        try:
            blocks_shape = self._irs_blocks_nm.shape
        except AttributeError:
            blocks_shape = '(None)'
        return '[{}, _irs_grid=len{}, _irs_blocks_nm=shape{}, _arir_config={}]'.format(
            super(FilterSetMiro, self).__str__()[1:-1], self._irs_grid.azimuth.shape[0], blocks_shape,
            self._arir_config.__str__().replace('\n   ', ''))

    def _log_load(self, logger=None):
        """
        Log convenient status information about the audio file being read. If no `logger` is provided the information
        will just be output via regular `print()`.

        Parameters
        ----------
        logger : logging.Logger, optional
            instance to provide identical logging behaviour as the parent process
        """
        log_str = 'opening file "{}"'.format(os.path.relpath(self._file_name))

        # generate file info
        array_signal = sfa.io.read_miro_struct(self._file_name)
        log_str += '\n --> samplerate: {:.0f} Hz, channels: {}, duration: {} samples, format: {}'.format(
            array_signal.signal.fs, array_signal.signal.signal.shape[0], array_signal.signal.signal.shape[1],
            array_signal.signal.signal.dtype)

        logger.info(log_str) if logger else print(log_str)

    def _load(self):
        """Load a filter set containing impulse responses for provided spherical grid positions."""
        if self._is_hrir:
            array_signal_l = sfa.io.read_miro_struct(self._file_name, channel='irChOne')
            array_signal_r = sfa.io.read_miro_struct(self._file_name, channel='irChTwo')
            self._irs_td = np.stack((array_signal_l.signal.signal, array_signal_r.signal.signal))
            self._irs_td = np.swapaxes(self._irs_td, 0, 1).copy()  # copy to ensure C-order

            # make sure provided sampling frequencies are identical and safe for reference
            assert array_signal_l.signal.fs == array_signal_r.signal.fs

            # make sure provided sampling grids are identical and safe for reference
            np.testing.assert_array_equal(array_signal_l.grid.azimuth, array_signal_r.grid.azimuth)
            np.testing.assert_array_equal(array_signal_l.grid.colatitude, array_signal_r.grid.colatitude)
            np.testing.assert_array_equal(array_signal_l.grid.radius, array_signal_r.grid.radius)
            np.testing.assert_array_equal(array_signal_l.grid.weight, array_signal_r.grid.weight)

            # select one set for further reference, since they are identical
            array_signal = array_signal_l

            self._points_azim_deg = array_signal.grid.azimuth * 360 / (2 * np.pi)
            self._points_elev_deg = array_signal.grid.colatitude * 360 / (2 * np.pi) + 90
        else:
            array_signal = sfa.io.read_miro_struct(self._file_name)
            self._irs_td = array_signal.signal.signal[np.newaxis, :]
            self._arir_config = array_signal.configuration

            # delete attributes if not needed
            del self._points_azim_deg
            del self._points_elev_deg

        # save needed attributes
        self._fs = int(array_signal.signal.fs)
        self._irs_grid = array_signal.grid

        # # shrink size of saved grid
        # self._irs_grid = sfa.io.SphericalGrid(self._irs_grid.azimuth.astype(np.float16),
        #                                       self._irs_grid.colatitude.astype(np.float16),
        #                                       self._irs_grid.radius.astype(np.float16),
        #                                       self._irs_grid.weight.astype(np.float32))

    def calculate_filter_blocks_nm(self, logger=None):
        """
        Transform beforehand calculated block-wise one-sided complex spectra in frequency domain into spherical
        harmonics coefficients according to the provided spatial structure by `_sh_max_order` and `_irs_grid`.

        Parameters
        ----------
        logger : logging.Logger, optional
            instance to provide identical logging behaviour as the parent process

        Raises
        ------
        RuntimeError
            in case frequency domain blocks have not been calculated yet
        """

        def _calculate_filter_block_nm(block_fd):
            """
            Parameters
            ----------
            block_fd : numpy.ndarray
                block of one-sided complex frequency spectra of size [number of filter points; number of output
                channels; block length (+1 depending on even or uneven length)]

            Returns
            -------
            numpy.ndarray
                block of complex spherical harmonics coefficients of size [number according to `sh_order`; number of
                output channels; block length (+1 depending on even or uneven length)]
            """
            # for each channel (second to last dimension)
            ir_nm = np.stack([sfa.process.spatFT(block_fd[:, ch, :], self._irs_grid, order_max=self._sh_max_order)
                              for ch in range(block_fd.shape[1])])
            return np.swapaxes(ir_nm, 0, 1)

        if self._irs_blocks_fd is None:
            raise RuntimeError('filter blocks in frequency domain have not been calculated yet.')

        # print warning if filter is multiple blocks long
        if self._irs_blocks_fd.shape[0] > 1:
            log_str = 'number of filter blocks in frequency domain is {} (greater then 1).'.format(
                self._irs_blocks_fd.shape[0])
            logger.warning(log_str) if logger else print('[WARNING]  ' + log_str, file=sys.stderr)

        # for each block
        self._irs_blocks_nm = np.stack([_calculate_filter_block_nm(block_fd) for block_fd in self._irs_blocks_fd])
        self._irs_blocks_nm = self._irs_blocks_nm.copy()  # copy to ensure C-order

    def apply_radial_filter(self, arir_config, amp_limit_db, is_apply_shift=True, logger=None):
        """
        Calculate a modal radial filter according to the provided input array configuration and the calculated
        coefficients will then be applied to the loaded HRIR coefficients.

        Parameters
        ----------
        arir_config : sfa.io.ArrayConfiguration
            recording / measurement microphone array configuration
        amp_limit_db : int
             maximum modal amplification limit in dB
        is_apply_shift : bool, optional
            if time shift of half block length should be applied
        logger : logging.Logger, optional
            instance to provide identical logging behaviour as the parent process

        Raises
        ------
        RuntimeError
            in case frequency domain blocks have not been calculated yet
        """
        if self._irs_blocks_nm is None:
            raise RuntimeError('filter blocks in spherical harmonics domain have not been calculated yet.')

        # calculate sh-coefficients
        sh_m, _ = sfa.sph.mnArrays(self._sh_max_order)
        sh_m_power = np.float_power(-1.0, sh_m)[:, np.newaxis, np.newaxis]
        nfft = self._dirac_td.shape[-1] * 2  # resulting radial filter will be identical size to already loaded filter
        radial_filter = sfa.gen.radial_filter_fullspec(max_order=self._sh_max_order, NFFT=nfft,
                                                       fs=self._fs, array_configuration=arir_config,
                                                       amp_maxdB=amp_limit_db)

        if is_apply_shift:
            shift_seconds = self._dirac_td.shape[-1] / (2 * self._fs)
            log_str = 'applying time shift of {:.0f}ms to radial filters ...'.format(shift_seconds * 1000)
            logger.info(log_str) if logger else print(log_str)

            # apply radial filter delay of a half block length
            radial_filter *= tools.generate_delay_fd(radial_filter.shape[-1], self._fs, shift_seconds)

        # plot radial filters TD and FD
        name = '{}_RadF_{:d}_sh{:d}_{:d}db'.format(
            logger.name, self._dirac_td.shape[-1], self._sh_max_order, amp_limit_db)
        tools.export_plot(tools.plot_ir_and_tf(radial_filter, self._fs, is_label_y=True, is_share_y=True),
                          name, logger=logger)

        # # plot radial filters TD around peak
        # _PEAK_FRAME_SAMPLES = 128
        # name = '{}_RadF_{:d}-{:d}_sh{:d}_{:d}db'.format(logger.name, self._dirac_td.shape[-1], _PEAK_FRAME_SAMPLES,
        #                                                 self._sh_max_order, amp_limit_db)
        # tools.export_plot(self._plot_peaks_framed(radial_filter, _PEAK_FRAME_SAMPLES), name, logger=logger)

        # repeat to match number of sh-coefficients
        radial_filter = np.repeat(radial_filter, range(1, self._sh_max_order * 2 + 2, 2), axis=0)
        radial_filter = np.repeat(radial_filter[:, np.newaxis, :], self._irs_blocks_nm.shape[-2], axis=1)

        # apply sh-coefficients for each block
        for b in range(self._irs_blocks_nm.shape[0]):
            self._irs_blocks_nm[b] *= radial_filter * sh_m_power

    @staticmethod
    def _plot_peaks_framed(data_fd, peak_frame_samples, n_cols=5):
        """
        Parameters
        ----------
        data_fd : numpy.ndarray
            frequency domain (complex) data that should be plotted
        peak_frame_samples : int
            number of samples that should be plotted IR around peak (reference is 0th order)
        n_cols : int, optional
            number of columns the plots should be organized in

        Returns
        -------
        matplotlib.figure.Figure
            generated plot
        """
        import matplotlib.pyplot as plt

        # transform into td
        data_td = np.fft.irfft(data_fd)

        # calculate peak sample and plot range
        peak_sample = np.abs(data_td[0]).argmax()
        peak_range = np.array([peak_sample, peak_sample + peak_frame_samples])
        peak_range = np.round(peak_range - min([peak_frame_samples / 2, peak_sample]))

        n_rows = int(np.ceil(data_td.shape[0] / n_cols))
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, squeeze=False, sharex='all', sharey='all')
        for ch in range(n_cols * n_rows):
            c = ch % n_cols
            r = int(ch / n_cols)

            if ch < data_td.shape[0]:
                axes[r, c].plot(data_td[ch], linewidth=.5)

            axes[r, c].set_xlim(*peak_range)
            axes[r, c].tick_params(direction='in', top=True, bottom=True, left=True, right=True)
            if r == n_rows - 1:
                axes[r, c].set_xlabel('Samples')
            if c == 0:
                axes[r, c].set_ylabel('Amplitude')

        # remove layout margins
        fig.tight_layout(pad=0)

        return fig

    def _get_index_from_rotation(self, azim_deg, elev_deg):
        """
        Parameters
        ----------
        azim_deg : int or float
            azimuth of desired sound incidence direction in degrees
        elev_deg: int or float
            elevation of desired sound incidence direction in degrees

        Returns
        -------
        int
            index in the `numpy.ndarray` storing the impulse response according to the desired incidence direction
        """
        # TODO: change to grid based search method of closest point

        # get closest azimuth index (catch case when index at end of vector would be detected)
        azim_ids = min(self._points_azim_deg.searchsorted(azim_deg, side='left' if azim_deg >= 0 else 'right'),
                       self._points_azim_deg.shape[0] - 1)

        # get indices of all azimuths having the same value
        azim_ids = np.where(self._points_azim_deg == self._points_azim_deg[azim_ids])[0]

        # get closest elevation index of subset (catch case when index at end of vector would be detected)
        elev_sub_id = min(self._points_elev_deg[azim_ids].searchsorted(elev_deg), azim_ids.shape[0] - 1)

        return azim_ids[elev_sub_id]

    def get_filter_blocks_fd(self, azim_deg=0.0, elev_deg=0.0):
        """
        Parameters
        ----------
        azim_deg : int or float, optional
            azimuth of desired sound incidence direction in degrees (only relevant for HRIR)
        elev_deg: int or float, optional
            elevation of desired sound incidence direction in degrees (only relevant for HRIR)

        Returns
        -------
        numpy.ndarray
            block-wise one-sided complex frequency spectra of the filters of size [number of blocks; blocksize (+1
            depending on even or uneven length); number of output channels; number of optional hrir positions]

        Raises
        ------
        RuntimeError
            in case requested blocks have not been calculated yet
        """
        if self._is_hrir:
            return super(FilterSetMiro, self).get_filter_blocks_fd(azim_deg, elev_deg)
        else:
            if self._irs_blocks_fd is None:
                raise RuntimeError('filter blocks in frequency domain have not been calculated yet.')

            return self._irs_blocks_fd

    def get_filter_blocks_nm(self):
        """
        Returns
        -------
        numpy.ndarray
            block-wise complex spherical harmonics coefficients of size [number of blocks; number according to
            `sh_order`; number of output channels; block length (+1 depending on even or uneven length)]

        Raises
        ------
        RuntimeError
            in case requested blocks have not been calculated yet
        """
        if self._irs_blocks_nm is None:
            raise RuntimeError('filter blocks in spherical harmonics domain have not been calculated yet.')

        return self._irs_blocks_nm

    def get_sh_configuration(self):
        """
        Returns
        -------
        FilterSetShConfig
            combined filter configuration with all necessary information to transform an incoming audio block into
            spherical harmonics sound field coefficients in real-time
        """
        # calculate sh orders
        sh_m, _ = sfa.sph.mnArrays(self._sh_max_order)

        # calculate sh base functions being weighted
        sh_bases = sfa.sph.sph_harm_all(self._sh_max_order, self._irs_grid.azimuth, self._irs_grid.colatitude)
        sh_bases_weighted = np.conj(sh_bases).T * (4 * np.pi * self._irs_grid.weight)

        return FilterSetShConfig(sh_m, sh_bases_weighted, self._arir_config)


class FilterSetShConfig(namedtuple('FilterSetShConfig', ['sh_m', 'sh_bases_weighted', 'arir_config'])):
    """
    Named tuple to combine information of a spherical filter set necessary (i.e. microphone array) for spherical
    harmonics processing of another filter set (i.e. HRIR).
    """
    __slots__ = ()

    # noinspection PyUnusedLocal
    def __init__(self, sh_m, sh_bases_weighted, arir_config):
        """
        Parameters
        ----------
        sh_m : numpy.ndarray
           set of spherical harmonics orders of size [number according to `sh_order`]
        sh_bases_weighted : numpy.ndarray
            spherical harmonic bases weighted by grid weights of spatial sampling points of size [number according to
            `sh_max_order`; number of input channels]
        arir_config : sfa.io.ArrayConfiguration
           recording / measurement microphone array configuration
        """
        super(FilterSetShConfig, self).__init__()

    def __new__(cls, sh_m, sh_bases_weighted, arir_config):
        # noinspection PyArgumentList
        return super(FilterSetShConfig, cls).__new__(cls, sh_m, sh_bases_weighted, arir_config)


class FilterSetSofa(FilterSet):
    def _load(self):
        """
        """
        raise NotImplementedError('chosen filter type "{}" is not implemented yet.'.format(type(self)))

    def _get_index_from_rotation(self, azim_deg, elev_deg):
        """
        """
        raise NotImplementedError('chosen filter type "{}" is not implemented yet.'.format(type(self)))
