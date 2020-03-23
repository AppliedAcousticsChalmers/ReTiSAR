import os
import sys
from collections import namedtuple
from enum import auto, Enum

import numpy as np
import pysofaconventions as sofa
import samplerate
import sound_field_analysis as sfa
import soundfile

from . import DataRetriever, tools


class FilterSet(object):
    """
    Flexible structure used to store FIR filter sets with an arbitrary number of channels. A
    given `FilterSet.Type` defines how individual input files are read and impulse responses are
    provided to a `Convolver` instance.

    All relevant time or frequency information of the contained filters are saved in
    multi-dimensional `numpy.ndarray`s for efficiency.

    Attributes
    ----------
    _file_name : str
        file path/name of filter source file
    _is_hrir : bool
        if filter set contains HRIR data
    _is_hpcf : bool
        if filter set contains HPCF data
    _points_azim_deg : numpy.ndarray
        azimuth angles in degrees of dedicated head rotations at which the set contains impulse
        responses
    _points_elev_deg : numpy.ndarray
        elevation angles in degrees of dedicated head rotations at which the set contains impulse
        responses
    _fs : int
        filter data sampling frequency
    _irs_td : numpy.ndarray
        filter impulse response in time domain of size [number of hrir positions; number of output
        channels; number of samples]
    _irs_blocks_fd : numpy.ndarray
        block-wise one-sided complex frequency spectra of the filters of size [number of blocks;
        number of hrir positions; number of output channels; block length (+1 depending on even or
        uneven length)]
    _dirac_td : numpy.ndarray
        dirac impulse in time domain of identical size like `_irs_td`
    _dirac_blocks_fd : numpy.ndarray
        block-wise one-sided complex frequency spectra of dirac impulses of identical size like
        `_irs_blocks_fd`
    _irs_orig_shape : tuple of int
        filter impulse response original shape, considering already performed transposes or
        adjustments alike and potentially resampling
    """

    class Type(Enum):
        """
        Enumeration data type used to get an identification of loaded digital audio filter sets.
        It's attributes (with an arbitrary distinct integer value) are used as system wide unique
        constant identifiers.
        """

        FIR_MULTICHANNEL = auto()
        """Impulse responses with an arbitrary number of channels used to convolve signals from
        input to output in a 1:1 relation.

        Used by `OverlapSaveConvolver` with size [fir_samples; channels].
        """

        HPCF_FIR = auto()
        """Headphone Compensation Filter in a shape of a regular WAV file or itaAudio as being
        specified in the ITA-Toolbox.
        http://ita-toolbox.org/

        Used by `OverlapSaveConvolver` with size [fir_samples; 2].
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
        Head Related Impulse Responses in a shape as being specified in the Measured Impulse
        Response Object.
        http://audiogroup.web.th-koeln.de/FILES/miro_documentation.pdf

        Used by `AdjustableFdConvolver` with flexible size specified by the measurement point grid.
        """

        ARIR_MIRO = auto()
        """
        Array Room Impulse Responses in a shape as being specified in the Measured Impulse Response
        Object.
        http://audiogroup.web.th-koeln.de/FILES/miro_documentation.pdf

        Used by `AdjustableFdConvolver` with flexible size specified by the measurement point grid.
        """

        AS_MIRO = auto()
        """
        Array audio stream with a configuration file as being specified in the Measured Impulse
        Response Object.
        http://audiogroup.web.th-koeln.de/FILES/miro_documentation.pdf

        Used by `AdjustableFdConvolver` with flexible size specified by the measurement point grid.
        """

        HRIR_SOFA = auto()
        """
        Head Related Impulse Responses in a shape as being specified in the Spatially Oriented
        Format for Acoustics under the convention "SimpleFreeFieldHRIR".
        https://www.sofaconventions.org/mediawiki/index.php/SOFA_(Spatially_Oriented_Format_for_Acoustics)

        Used by `AdjustableFdConvolver` with flexible size specified by the measurement point
        grid. Since there is no possibility to specify spatial grid weights, they are calculated
        by a pseudo inverse of the spherical harmonics base functions,
        see `sound_field_analysis-py` for reference! """

        ARIR_SOFA = auto()
        """Array Room Impulse Responses in a shape as being specified in the Spatially Oriented
        Format for Acoustics under the convention "SingleRoomDRIR".
        https://www.sofaconventions.org/mediawiki/index.php/SOFA_(Spatially_Oriented_Format_for_Acoustics)

        Used by `AdjustableFdConvolver` with flexible size specified by the measurement point
        grid. Since there is no possibility to specify spatial grid weights, they are calculated
        by a pseudo inverse of the spherical harmonics base functions,
        see `sound_field_analysis-py` for reference! """

    _ERROR_MSG_FD = "blocks in frequency domain have not been calculated yet."
    _ERROR_MSG_NM = "blocks in spherical harmonics domain have not been calculated yet."

    @staticmethod
    def create_instance_by_type(
        file_name, file_type, sh_max_order=None, sh_is_enforce_pinv=False
    ):
        """
        Parameters
        ----------
        file_name : str, numpy.ndarray or None
            file path/name of filter source file or directly provided filter coefficients
        file_type : str, FilterSet.Type or None
            type of filter source file
        sh_max_order : int, optional
            spherical harmonics order used for the spatial Fourier transform
        sh_is_enforce_pinv : bool, optional
            if pseudo-inverse (Moore-Penrose) matrix will be used over explicitly given sampling
            grid weights to calculate the weighted SH basis functions, only relevant for filter
            sets in MIRO format

        Returns
        -------
        FilterSetMultiChannel, FilterSetSsr, FilterSetMiro FilterSetSofa or None
            created instance according to `FilterSet.Type`
        """
        _type = tools.transform_into_type(file_type, FilterSet.Type)
        if _type is None or file_name is None:
            return None
        elif isinstance(file_name, str) and (
            file_name.strip("'\"") == "" or file_name.upper() == "NONE"
        ):
            return None
        elif _type == FilterSet.Type.FIR_MULTICHANNEL:
            return FilterSetMultiChannel(file_name=file_name, is_hrir=False)
        elif _type == FilterSet.Type.HPCF_FIR:
            return FilterSetMultiChannel(
                file_name=file_name, is_hrir=False, is_hpcf=True
            )
        elif _type == FilterSet.Type.HRIR_SSR:
            return FilterSetSsr(file_name=file_name, is_hrir=True)
        elif _type == FilterSet.Type.BRIR_SSR:
            return FilterSetSsr(file_name=file_name, is_hrir=False)
        elif _type == FilterSet.Type.HRIR_MIRO:
            return FilterSetMiro(
                file_name=file_name,
                is_hrir=True,
                sh_max_order=sh_max_order,
                sh_is_enforce_pinv=sh_is_enforce_pinv,
            )
        elif _type in [FilterSet.Type.ARIR_MIRO, FilterSet.Type.AS_MIRO]:
            return FilterSetMiro(
                file_name=file_name,
                is_hrir=False,
                sh_max_order=sh_max_order,
                sh_is_enforce_pinv=sh_is_enforce_pinv,
            )
        elif _type == FilterSet.Type.HRIR_SOFA:
            return FilterSetSofa(
                file_name=file_name,
                is_hrir=True,
                sh_max_order=sh_max_order,
                sh_is_enforce_pinv=sh_is_enforce_pinv,
            )
        elif _type == FilterSet.Type.ARIR_SOFA:
            return FilterSetSofa(
                file_name=file_name,
                is_hrir=False,
                sh_max_order=sh_max_order,
                sh_is_enforce_pinv=sh_is_enforce_pinv,
            )

    def __init__(self, file_name, is_hrir, is_hpcf=False):
        """
        Initialize FIR filter, call `load()` afterwards to load the file contents!

        Parameters
        ----------
        file_name : str
            file path/name of filter source file
        is_hrir : bool
            if filter set contains HRIR data
        is_hpcf : bool, optional
            if filter set contains HPCF data
        """
        self._file_name = file_name
        self._is_hrir = is_hrir
        self._is_hpcf = is_hpcf

        self._points_azim_deg = None
        self._points_elev_deg = None
        self._fs = None
        self._irs_td = None
        self._irs_blocks_fd = None
        self._dirac_td = None
        self._dirac_blocks_fd = None
        self._irs_orig_shape = None

    def __str__(self):
        try:
            grid = (
                f"_points_azim_deg=shape{self._points_azim_deg.shape}, "
                f"_points_elev_deg=shape{self._points_elev_deg.shape}, "
            )
        except AttributeError:
            grid = ""
        return (
            f"[ID={id(self)}, _file_name={os.path.relpath(self._file_name)}, {grid}_fs={self._fs}, "
            f"_irs_td=shape{self._irs_td.shape}, _irs_blocks_fd=shape{self._irs_blocks_fd.shape}]"
        )

    def load(
        self,
        block_length,
        is_single_precision,
        logger=None,
        ir_trunc_db=None,
        check_fs=None,
        is_prevent_resampling=False,
        is_prevent_logging=False,
        is_normalize=False,
        is_normalize_individually=False,
    ):
        """
        Loading the file contents of the provided FIR filter according to its specification. The
        loading procedures are are provided by the individual subclasses. This function also
        generates `_dirac_td` with identical size as `_irs_td`.

        Parameters
        ----------
        block_length : int or None
            system specific size of every audio block
        is_single_precision : bool
            if data should be stored and according processing done in single precision for better
            performance, double precision otherwise
        logger : logging.Logger, optional
            instance to provide identical logging behaviour as the parent process
        ir_trunc_db : float, optional
            impulse response truncation level in dB relative under peak
        check_fs : int, optional
            global system sampling frequency, which is checked to be identical with or otherwise
            the loaded filter will be resampled  (see `is_prevent_resampling`)
        is_prevent_resampling : bool, optional
            if loaded filter impulse response should not be resampled
        is_prevent_logging : bool, optional
            prevent logging messages during load process
        is_normalize : bool, optional
            if loaded filter impulse response should be normalized regarding its global peak
        is_normalize_individually : bool, optional
            if each channel of the loaded filter impulse response should be normalized regarding
            its individual peak (except for second to last dimension, resembling binaural pairs of
            left and right ear)
        """
        # gather data, in case local file does not exist and online reference is given
        self._file_name = DataRetriever.retrieve(path=self._file_name, logger=logger)

        if not is_prevent_logging:
            self._log_load(logger=logger)

        # type specific loading
        self._load(dtype=np.float32 if is_single_precision else np.float64)
        # delete attributes if not needed
        if not self._is_hrir:
            del self._points_azim_deg
            del self._points_elev_deg

        # truncate if requested
        if ir_trunc_db:
            self._truncate(
                logger=logger, cutoff_db=ir_trunc_db, block_length=block_length
            )

        # check sampling frequency match or initialize resampling if requested
        if not self._fs and check_fs:
            self._fs = check_fs
        elif check_fs and check_fs != self._fs:
            self._resample(
                logger=logger,
                target_fs=check_fs,
                is_prevent_resampling=is_prevent_resampling,
            )

        # normalize if requested
        if is_normalize_individually or is_normalize:
            self._normalize(logger=logger, is_individually=is_normalize_individually)

        # store original filter length
        self._irs_orig_shape = self._irs_td.shape
        # zero-padding if necessary
        self._zero_pad(block_length=block_length)

        if not is_prevent_logging:
            # plot filters TD and FD
            if self._is_hrir or self._is_hpcf or type(self) == FilterSetSsr:
                data = self._irs_td[0]
            else:
                data = self._irs_td[:, :8]

            if isinstance(self._file_name, np.ndarray):
                # extract generated size
                name = f'GENERATED,{str(self._file_name.shape).strip("()").replace(" ", "")}'
            else:
                # extract parent dir
                top_dir = os.path.basename(os.path.dirname(self._file_name))
                # extract file name without ending
                name = os.path.basename(self._file_name).rsplit(".")[0]
                name = f"{top_dir}_{name}"
            name = f"{logger.name if logger else self.__module__}_{name}_{block_length}"

            tools.export_plot(
                figure=tools.plot_ir_and_tf(
                    data_td_or_fd=data,
                    fs=self._fs,
                    set_fd_db_y=30,
                    set_td_db_y=120,
                    is_etc=True,
                    step_db_y=10,
                ),
                name=name,
                logger=logger,
            )

        # generate dirac impulse
        self._dirac_td = np.zeros(self._irs_td.shape[-2:], dtype=self._irs_td.dtype)
        self._dirac_td[:, 0] = 1.0

    def _log_load(self, logger=None):
        """
        Log convenient status information about the audio file being read. If no `logger` is
        provided the information will just be output via regular `print()`.

        Parameters
        ----------
        logger : logging.Logger, optional
            instance to provide identical logging behaviour as the parent process
        """
        if isinstance(self._file_name, np.ndarray):
            log_str = f"loading provided filter coefficients with size {self._file_name.shape}"
        else:
            log_str = f'opening file "{os.path.relpath(self._file_name)}"'

            try:
                # cut off name and reformat LF
                file_info = (
                    soundfile.info(self._file_name)
                    .__str__()
                    .split("\n", 1)[1]
                    .replace("\n", ", ")
                )
            except RuntimeError:
                try:
                    file_info = (
                        f"unable to open with `soundfile`, "
                        f"size: {os.stat(self._file_name).st_size / 1E6:.2f} MB"
                    )
                except FileNotFoundError:
                    raise ValueError(f"{log_str}\n --> file not found")

            log_str = f"{log_str}\n --> {file_info}"

        logger.info(log_str) if logger else print(log_str)

    def _load(self, dtype):
        """
        Individual implementation of loading the FIR filter according to its specifications.

        Parameters
        ----------
        dtype : str or numpy.dtype or type
            numpy data type the contents should be stored in
        """
        raise NotImplementedError(
            f'chosen filter type "{type(self)}" is not implemented yet.'
        )

    def _truncate(self, logger, cutoff_db, block_length):
        """
        Truncate loaded filter at the given cutoff level. The new length is calculated by all
        filter channels having decayed to the provided level relative under global peak.

        Parameters
        ----------
        logger : logging.Logger
            instance to provide identical logging behaviour as the parent process
        cutoff_db : float
            truncation level in dB relative under global peak

        Raises
        ------
        ValueError
            in case truncation cutoff level is greater than 0
        """
        if not cutoff_db:
            return
        if cutoff_db > 0:
            raise ValueError("data IR truncation level of greater than 0 given.")

        # tools.plot_ir_and_tf(self._irs_td[0, 0], self._fs, is_etc=True, is_show_blocked=False)

        # calculate ETC
        data_td = self._irs_td.copy()
        data_td[np.nonzero(data_td == 0)] = 1e-12  # prevent errors
        etc = 20 * np.log10(np.abs(data_td))

        # find last index that is over provided threshold for ever channel
        # adjust threshold to cutoff value under global peak
        cutoff_index = np.nonzero(etc > cutoff_db + etc.max())
        len_max = cutoff_index[-1].max() + 1
        if len_max >= self._irs_td.shape[-1]:
            return
        # adjust length to fill entire blocks
        len_adj = ((len_max // block_length) + 1) * block_length
        if len_adj >= self._irs_td.shape[-1]:
            return

        def _get_channel_str(c1, c2):
            if not c1:
                return f"{c2}"
            elif not c2:
                return f"{c1}"
            else:
                return f"[{c1}, {c2}]"

        # log truncation lengths
        i_max = cutoff_index[-1].argmax()
        log_str = (
            f'truncating data length from "{os.path.relpath(self._file_name)}"'
            f"\n --> source: {data_td.shape[-1]} samples, cutoff: {cutoff_db} dB, "
            f"result: {len_max} samples (decay on channel "
            f"{_get_channel_str(cutoff_index[0][i_max], cutoff_index[1][i_max])})"
        )
        if len_adj != len_max:
            log_str = f"{log_str}, block adjusted: {len_adj} samples"
        logger.warning(log_str) if logger else print(
            f"[WARNING]  {log_str}", file=sys.stderr
        )

        # truncate IRs
        self._irs_td = data_td[:, :, :len_max].copy()
        # tools.plot_ir_and_tf(self._irs_td[0, 0], self._fs, is_etc=True, is_show_blocked=False)

    def _resample(self, logger, target_fs, is_prevent_resampling=False):
        """
        Resample loaded filter to the given target sampling rate, utilizing the API of package
        `samplerate` to the C-library `libsamplerate`.

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
        _RESAMPLE_CONVERTER_TYPE = "sinc_best"
        """Resampling converter type, see `samplerate` library."""

        if self._fs == target_fs:
            return

        log_str = (
            f'resampling data from "{os.path.relpath(self._file_name)}"'
            f"\n --> source: {self._fs} Hz, target: {target_fs} Hz"
        )
        if is_prevent_resampling:
            raise ValueError(f"prevented {log_str}")
        logger.warning(log_str) if logger else print(
            f"[WARNING]  {log_str}", file=sys.stderr
        )

        # initialize target variables
        ratio_fs = target_fs / self._fs

        if self._irs_td.ndim == 3:
            # iteration over first dimension, since `samplerate.resample()` only takes
            # 2-dimensional inputs
            irs_td_new = np.stack(
                [
                    samplerate.resample(irs_td.T, ratio_fs, _RESAMPLE_CONVERTER_TYPE)
                    for irs_td in self._irs_td
                ]
            )
            # restore former dimension order
            irs_td_new = np.swapaxes(irs_td_new, 1, 2).copy()  # copy to ensure C-order
        else:
            raise NotImplementedError(
                f"resampling for {self._irs_td.ndim} ndarray dimensions not implemented yet."
            )

        # import matplotlib.pyplot as plt
        #
        # tools.plot_ir_and_tf(self._irs_td[0], self._fs)
        # plt.show(block=False)
        # tools.plot_ir_and_tf(irs_td_new[0], target_fs)
        # plt.show(block=False)

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
            if all channels should be normalized independent of each other (except for second to
            last dimension, resembling binaural pairs of left and right ear)
        target_amp : float, optional
            target amplitude that should be scaled to
        """
        if is_individually:
            irs_peak = (
                np.abs(self._irs_td)
                .max(axis=-1)
                .max(axis=-1)[:, np.newaxis, np.newaxis]
            )
            log_str = (
                f"[INFO]  normalized {len(irs_peak):d} IR peaks channel independent to "
                f"{target_amp:.2f}."
            )
            logger.warning(log_str) if logger else print(log_str, file=sys.stderr)
        else:
            irs_peak = np.abs(self._irs_td).max()
            log_str = f"normalized IR peak from {irs_peak:.2f} to {target_amp:.2f}."
            logger.info(log_str) if logger else print(log_str)

        self._irs_td *= target_amp / irs_peak

    def _zero_pad(self, block_length):
        """
        Append zeros in time domain to `_irs_td` to full blocks and if filter length is smaller
        then provided block length.

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
        irs_td_padded = np.zeros(
            (self._irs_td.shape[0], self._irs_td.shape[1], block_length * block_count)
        )
        irs_td_padded[:, :, : self._irs_td.shape[2]] = self._irs_td
        self._irs_td = irs_td_padded.astype(self._irs_td.dtype)  # `astype()` makes copy

    def calculate_filter_blocks_fd(self, block_length):
        """
        Split up the time domain information of the filter into blocks according to the provided
        length. This operation will add an additional dimension to the beginning of the
        block-wise `numpy.ndarray`s. For convenience in later processing the blocks will also be
        independently transformed into frequency domain and saved as complex one-sided spectra.

        The transformation process is also emulated for the dirac impulse response data. This is
        saved into `_dirac_blocks_fd` with identical size as `_irs_blocks_fd`.

        Parameters
        ----------
        block_length : int or None
            system wide length of audio blocks in samples
        """
        # entire signal is one block, if no size is given
        if not block_length:  # None or <=0
            block_length = self._irs_td.shape[-1]

        # for filter
        block_count = self._irs_td.shape[-1] // block_length
        block_length_2 = block_length * 2
        # cut signal into slices and stack on new axis after transformation in frequency domain
        self._irs_blocks_fd = np.stack(
            [
                np.fft.rfft(block_td, block_length_2)
                for block_td in np.dsplit(self._irs_td, block_count)
            ]
        )
        # rfft not replaced by `pyfftw` since it is not executed in real-time

        # numpy `rfft()` will always yield double precision!!
        if self._irs_td.dtype == np.float32:
            self._irs_blocks_fd = self._irs_blocks_fd.astype(
                np.complex64
            )  # `astype()` makes copy

        # for dirac impulse, limited to be rotation independent
        self._dirac_blocks_fd = np.zeros_like(self._irs_blocks_fd)[:, :1]
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
            raise RuntimeError(FilterSet._ERROR_MSG_FD)
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
            filter impulse response in time domain of size [number of output channels;
            number of samples]
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
            block-wise one-sided complex frequency spectra of the filters of size
            [number of blocks; number of output channels; block length (+1 depending on even or
            uneven length)]

        Raises
        ------
        RuntimeError
            in case requested blocks have not been calculated yet
        """
        if self._irs_blocks_fd is None:
            raise RuntimeError(FilterSet._ERROR_MSG_FD)

        index = self._get_index_from_rotation(azim_deg, elev_deg)
        # print(f'azim {azim_deg:>-4.0f}, elev {elev_deg:>-4.0f} -> index {index:>4.0f}')
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
            index in the `numpy.ndarray` storing the impulse response according to the desired
            incidence direction
        """
        raise NotImplementedError(
            f'chosen filter type "{type(self)}" is not implemented yet.'
        )


class FilterSetMultiChannel(FilterSet):
    """
    Flexible structure used to store FIR filter sets with an arbitrary number of channels.

    All relevant time or frequency information of the contained filters are saved in
    multi-dimensional `numpy.ndarray`s for efficiency.
    """

    def _load(self, dtype):
        """
        Load an arbitrary number of impulse responses from multi-channel audio file. There is no
        directional information associated with this data. This can be used to provide a
        multi-channel-wise convolution i.e., to equalize reproduction setups (headphones or
        loudspeakers).

        Parameters
        ----------
        dtype : str or numpy.dtype or type
            numpy data type the contents should be stored in

        Raises
        ------
        TypeError
            in case unknown type for filter name / impulse response is given
        """
        if isinstance(self._file_name, str):
            self._irs_td, self._fs = soundfile.read(self._file_name, dtype=dtype)
            self._irs_td = self._irs_td.T  # proper memory alignment will be done later
        elif isinstance(self._file_name, np.ndarray):
            self._irs_td = self._file_name
            self._fs = None
        else:
            raise TypeError(f'unknown parameter type "{type(self._file_name)}".')

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
            block-wise one-sided complex frequency spectra of the filters of size
            [number of blocks; 1; number of output channels; block length (+1 depending on even
            or uneven length)]

        Raises
        ------
        RuntimeError
            in case requested blocks have not been calculated yet
        """
        if self._irs_blocks_fd is None:
            raise RuntimeError(FilterSet._ERROR_MSG_FD)

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
            index in the `numpy.ndarray` storing the impulse response according to the desired
            incidence direction
        """
        return 0


class FilterSetSsr(FilterSet):
    """
    Flexible structure used to store Head Related Impulse Responses in a shape as being shipped
    with the SoundScapeRenderer.

    All relevant time or frequency information of the contained filters are saved in
    multi-dimensional `numpy.ndarray`s for efficiency.
    """

    def _load(self, dtype):
        """
        Load a filter set containing Head-Related Impulse Responses of 360 sources in equal
        distance on the horizontal plane. The file contains 720 audio channels, since the
        according IRs for each ear are provided consecutively for each source position.

        Parameters
        ----------
        dtype : str or numpy.dtype or type
            numpy data type the contents should be stored in

        Raises
        ------
        TypeError
            in case unknown type for filter name is given
        """
        if not isinstance(self._file_name, str):
            raise TypeError(f'unknown parameter type "{type(self._file_name)}".')

        irs_td, self._fs = soundfile.read(self._file_name, dtype=dtype)
        irs_td_l = irs_td[:, 0:-1:2].T  # all even elements
        irs_td_r = irs_td[:, 1::2].T  # all uneven elements
        self._irs_td = np.swapaxes(
            np.stack((irs_td_l, irs_td_r)), 0, 1
        )  # proper memory alignment will be done later

        self._points_azim_deg = np.linspace(
            0, 360, self._irs_td.shape[0], endpoint=False, dtype=np.int16
        )
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
            index in the `numpy.ndarray` storing the impulse response according to the desired
            incidence direction
        """
        # invert left handed orientation of SSR grid and round down
        return int(-azim_deg)


class FilterSetMiro(FilterSet):
    """
    Flexible structure used to store Head Related Impulse Responses or Array Room Impulse
    Responses in a shape as being specified in the Measured Impulse Response Object.

    All relevant time, frequency or spherical harmonics information of the contained filters are
    saved in multi-dimensional `numpy.ndarray`s for efficiency. Relevant spatial sampling grid
    information is saved in structures provided by `sound-field-analysis-py`.

    Attributes
    ----------
    _sh_max_order : int
        maximum spherical harmonics order used for the spatial Fourier transform
    _sh_is_enforce_pinv : bool
        if pseudo-inverse (Moore-Penrose) matrix will be used over explicitly given sampling grid
        weights to calculate the weighted SH basis functions, only relevant for filter sets in
        MIRO format
    _irs_grid : sfa.io.SphericalGrid
        recording / measurement point configuration
    _irs_blocks_nm : numpy.ndarray
        block-wise complex spherical harmonics coefficients of size [number of blocks; number
        according to `sh_order`; number of output channels; block length (+1 depending on even or
        uneven length)]
    _arir_config : sfa.io.ArrayConfiguration
        recording / measurement microphone array configuration
    """

    def __init__(self, file_name, is_hrir, sh_max_order=None, sh_is_enforce_pinv=False):
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
        sh_is_enforce_pinv : bool, optional
            if pseudo-inverse (Moore-Penrose) matrix will be used over explicitly given sampling
            grid weights to calculate the weighted SH basis functions, only relevant for filter
            sets in MIRO format
        """
        super().__init__(file_name=file_name, is_hrir=is_hrir)
        self._sh_max_order = sh_max_order
        self._sh_is_enforce_pinv = sh_is_enforce_pinv

        self._irs_grid = None
        self._irs_blocks_nm = None
        self._arir_config = None

    def __str__(self):
        try:
            blocks_shape = self._irs_blocks_nm.shape
        except AttributeError:
            blocks_shape = "(None)"

        arir_str = self._arir_config.__str__().replace("\n   ", "")
        return (
            f"[{super().__str__()[1:-1]}, _sh_max_order={self._sh_max_order}, "
            f"_sh_is_enforce_pinv={self._sh_is_enforce_pinv}, "
            f"_irs_grid=len{self._irs_grid.azimuth.shape[0]}, "
            f"_irs_blocks_nm=shape{blocks_shape}, _arir_config={arir_str}]"
        )

    def _log_load(self, logger=None):
        """
        Log convenient status information about the audio file being read. If no `logger` is
        provided the information will just be output via regular `print()`.

        Parameters
        ----------
        logger : logging.Logger, optional
            instance to provide identical logging behaviour as the parent process
        """
        log_str = f'opening file "{os.path.relpath(self._file_name)}"'

        # generate file info
        try:
            array_signal = sfa.io.read_miro_struct(self._file_name)
            log_str = (
                f"{log_str}\n --> samplerate: {array_signal.signal.fs:.0f} Hz, "
                f"channels: {array_signal.signal.signal.shape[0]}, "
                f"duration: {array_signal.signal.signal.shape[1]} samples, "
                f"format: {array_signal.signal.signal.dtype}"
            )
            logger.info(log_str) if logger else print(log_str)
        except ValueError as e:
            raise ValueError(f"{log_str}\n --> {e.args[0]}")

    def _load(self, dtype):
        """
        Load a filter set containing impulse responses for provided spherical grid positions.

        Parameters
        ----------
        dtype : str or numpy.dtype or type
            numpy data type the contents should be stored in

        Raises
        ------
        TypeError
            in case unknown type for filter name is given
        """
        if not isinstance(self._file_name, str):
            raise TypeError(f'unknown parameter type "{type(self._file_name)}".')

        if self._is_hrir:
            array_signal_l = sfa.io.read_miro_struct(self._file_name, channel="irChOne")
            array_signal_r = sfa.io.read_miro_struct(self._file_name, channel="irChTwo")
            self._irs_td = np.stack(
                (array_signal_l.signal.signal, array_signal_r.signal.signal)
            )
            # proper memory alignment will be done later
            self._irs_td = np.swapaxes(self._irs_td, 0, 1)

            # make sure provided sampling frequencies are identical and safe for reference
            assert array_signal_l.signal.fs == array_signal_r.signal.fs

            # make sure provided sampling grids are identical and safe for reference
            np.testing.assert_array_equal(
                array_signal_l.grid.azimuth, array_signal_r.grid.azimuth
            )
            np.testing.assert_array_equal(
                array_signal_l.grid.colatitude, array_signal_r.grid.colatitude
            )
            np.testing.assert_array_equal(
                array_signal_l.grid.radius, array_signal_r.grid.radius
            )
            np.testing.assert_array_equal(
                array_signal_l.grid.weight, array_signal_r.grid.weight
            )

            # select one set for further reference, since they are identical
            array_signal = array_signal_l

            self._points_azim_deg = np.rad2deg(array_signal.grid.azimuth.astype(dtype))
            # transform colatitude into elevation!
            self._points_elev_deg = 90 - np.rad2deg(
                array_signal.grid.colatitude.astype(dtype)
            )

        else:
            array_signal = sfa.io.read_miro_struct(self._file_name)

            # save needed attributes and adjust dtype
            self._irs_td = array_signal.signal.signal[np.newaxis, :]
            self._arir_config = sfa.io.ArrayConfiguration(
                *(
                    a.astype(dtype) if isinstance(a, np.ndarray) else a
                    for a in array_signal.configuration
                )
            )

        # save needed attributes and adjust dtype
        self._irs_td = self._irs_td.astype(dtype).copy()
        self._fs = int(array_signal.signal.fs)
        self._irs_grid = sfa.io.SphericalGrid(
            *(
                g.astype(dtype) if isinstance(g, np.ndarray) else g
                for g in array_signal.grid
            )
        )  # all values with correct dtype

        # # overwrite grid weights with ones just for testing
        # self._irs_grid = sfa.io.SphericalGrid(
        #     azimuth=self._irs_grid.azimuth,
        #     colatitude=self._irs_grid.colatitude,
        #     radius=self._irs_grid.radius,
        #     weight=np.ones_like(self._irs_grid.weight) / len(self._irs_grid.weight),
        # )

        # # print debug information
        # print(
        #     f"loaded grid.azimuth    {np.rad2deg(self._irs_grid.azimuth.min()):+6.1f} .. "
        #     f"{np.rad2deg(self._irs_grid.azimuth.max()):+6.1f} deg"
        # )
        # print(
        #     f"loaded grid.colatitude {np.rad2deg(self._irs_grid.colatitude.min()):+6.1f} .. "
        #     f"{np.rad2deg(self._irs_grid.colatitude.max()):+6.1f} deg"
        # )
        # if self._irs_grid.weight is None:
        #     print("loaded grid.weight       NONE")
        # else:
        #     print(
        #         f"loaded grid.weight   {self._irs_grid.weight.min():.6f} .. "
        #         f"{self._irs_grid.weight.max():.6f}"
        #     )

    def calculate_filter_blocks_nm(self):
        """
        Transform beforehand calculated block-wise one-sided complex spectra in frequency domain
        into spherical harmonics coefficients according to the provided spatial structure by
        `_sh_max_order` and `_irs_grid`.

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
                block of one-sided complex frequency spectra of size [number of filter points;
                number of output channels; block length (+1 depending on even or uneven length)]

            Returns
            -------
            numpy.ndarray
                block of complex spherical harmonics coefficients of size [number according to
                `sh_order`; number of output channels; block length (+1 depending on even or
                uneven length)]
            """
            # for each channel (second to last dimension)
            ir_nm = np.stack(
                [
                    sfa.process.spatFT_RT(
                        data=block_fd[:, ch, :],
                        spherical_harmonic_weighted=sh_bases_weighted,
                    )
                    for ch in range(block_fd.shape[1])
                ]
            )
            return np.swapaxes(ir_nm.astype(block_fd.dtype), 0, 1)

        if self._irs_blocks_fd is None:
            raise RuntimeError(FilterSet._ERROR_MSG_FD)

        # print warning if filter is multiple blocks long
        if self._irs_blocks_fd.shape[0] > 1:
            raise NotImplementedError(
                f"More than one ({self._irs_blocks_fd.shape[0]}) blocks would be necessary for "
                f"loaded filter and partitioned convolution in SH domain is not implemented yet."
            )

        # precompute weighted SH basis function
        sh_bases_weighted = self.get_sh_configuration().sh_bases_weighted

        # for each block
        self._irs_blocks_nm = np.stack(
            [_calculate_filter_block_nm(block_fd) for block_fd in self._irs_blocks_fd]
        )
        self._irs_blocks_nm = self._irs_blocks_nm.copy()  # copy to ensure C-order

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
            index in the `numpy.ndarray` storing the impulse response according to the desired
            incidence direction
        """
        # TODO: change to grid based search method of closest point

        # get closest azimuth index (catch case when index at end of vector would be detected)
        azim_ids = min(
            self._points_azim_deg.searchsorted(
                azim_deg, side="left" if azim_deg >= 0 else "right"
            ),
            self._points_azim_deg.shape[0] - 1,
        )

        # get indices of all azimuths having the same value
        azim_ids = np.where(self._points_azim_deg == self._points_azim_deg[azim_ids])[0]

        # get closest elevation index of subset (catch case when index at end of vector would be
        # detected)
        elev_sub_id = min(
            self._points_elev_deg[azim_ids].searchsorted(elev_deg),
            azim_ids.shape[0] - 1,
        )

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
            block-wise one-sided complex frequency spectra of the filters of size
            [number of blocks; blocksize (+1
            depending on even or uneven length); number of output channels;
            number of optional hrir positions]

        Raises
        ------
        RuntimeError
            in case requested blocks have not been calculated yet
        """
        if self._is_hrir:
            return super().get_filter_blocks_fd(azim_deg, elev_deg)
        else:
            if self._irs_blocks_fd is None:
                raise RuntimeError(FilterSet._ERROR_MSG_FD)

            return self._irs_blocks_fd

    def get_filter_blocks_nm(self):
        """
        Returns
        -------
        numpy.ndarray
            block-wise complex spherical harmonics coefficients of size [number of blocks;
            number according to `sh_order`; number of output channels; block length (+1 depending
            on even or uneven length)]

        Raises
        ------
        RuntimeError
            in case requested blocks have not been calculated yet
        """
        if self._irs_blocks_nm is None:
            raise RuntimeError(FilterSet._ERROR_MSG_NM)

        return self._irs_blocks_nm

    def get_sh_configuration(self):
        """
        Returns
        -------
        FilterSetShConfig
            combined filter configuration with all necessary information to transform an incoming
            audio block into spherical harmonics sound field coefficients in real-time

        Raises
        ------
        ValueError
            in case an invalid value for the maximal spherical harmonics processing order is given
        """
        if self._sh_max_order is None or self._sh_max_order < 0:
            raise ValueError(
                f'Invalid value "{self._sh_max_order}" for spherical harmonics order.'
            )

        # adjust dtype
        dtype = np.complex64 if self._irs_td.dtype == np.float32 else np.complex128

        # calculate sh orders
        sh_m = sfa.sph.mnArrays(self._sh_max_order)[0].astype(np.int16)

        # calculate sh base functions, see `sound-field-analysis-py` `process.spatFT()` for
        # reference!
        sh_bases = sfa.sph.sph_harm_all(
            self._sh_max_order, self._irs_grid.azimuth, self._irs_grid.colatitude
        ).astype(dtype)
        if self._sh_is_enforce_pinv or self._irs_grid.weight is None:
            # calculate pseudo inverse since no grid weights are given
            sh_bases_weighted = np.linalg.pinv(sh_bases)
        else:
            # apply given grid weights
            sh_bases_weighted = np.conj(sh_bases).T * (
                4 * np.pi * self._irs_grid.weight
            )

        return FilterSetShConfig(sh_m, sh_bases_weighted, self._arir_config)


class FilterSetShConfig(
    namedtuple("FilterSetShConfig", ["sh_m", "sh_bases_weighted", "arir_config"])
):
    """
    Named tuple to combine information of a spherical filter set necessary (i.e. microphone
    array) for spherical harmonics processing of another filter set (i.e. HRIR).
    """

    __slots__ = ()

    def __new__(cls, sh_m, sh_bases_weighted, arir_config):
        """
        Parameters
        ----------
        sh_m : numpy.ndarray
           set of spherical harmonics orders of size [number according to `sh_order`]
        sh_bases_weighted : numpy.ndarray
            spherical harmonic bases weighted by grid weights of spatial sampling points of size
            [number according to `sh_max_order`; number of input channels]
        arir_config : sfa.io.ArrayConfiguration
           recording / measurement microphone array configuration
        """
        # noinspection PyArgumentList
        return super().__new__(cls, sh_m, sh_bases_weighted, arir_config)

    def __str__(self):
        arir_str = self.arir_config.__str__().replace("\n   ", "")
        return (
            f"[sh_m=shape{self.sh_m.shape}, sh_bases_weighted=shape{self.sh_bases_weighted.shape}, "
            f"arir_config={arir_str}]"
        )


class FilterSetSofa(FilterSetMiro):
    """
    Flexible structure used to store Head Related Impulse Responses or Array Room Impulse
    Responses in a shape as being specified in the Spatially Oriented Format for Acoustics.

    All relevant time, frequency or spherical harmonics information of the contained filters are
    saved in multi-dimensional `numpy.ndarray`s for efficiency. Relevant spatial sampling grid
    information is saved in structures provided by `sound-field-analysis-py`.
    """

    def _log_load(self, logger=None):
        """
        Log convenient status information about the audio file being read. If no `logger` is
        provided the information will just be output via regular `print()`.

        Parameters
        ----------
        logger : logging.Logger, optional
            instance to provide identical logging behaviour as the parent process
        """

        def _load_convention(file):
            convention = file.getGlobalAttributeValue("SOFAConventions")
            if convention == "AmbisonicsDRIR":
                return sofa.SOFAAmbisonicsDRIR(file.ncfile.filename, "r")
            elif convention == "GeneralFIR":
                return sofa.SOFAGeneralFIR(file.ncfile.filename, "r")
            elif convention == "GeneralFIRE":
                return sofa.SOFAGeneralFIRE(file.ncfile.filename, "r")
            elif convention == "GeneralTF":
                return sofa.SOFAGeneralTF(file.ncfile.filename, "r")
            elif convention == "MultiSpeakerBRIR":
                return sofa.SOFAMultiSpeakerBRIR(file.ncfile.filename, "r")
            elif convention == "SimpleFreeFieldHRIR":
                return sofa.SOFASimpleFreeFieldHRIR(file.ncfile.filename, "r")
            elif convention == "SimpleFreeFieldSOS":
                return sofa.SOFASimpleFreeFieldSOS(file.ncfile.filename, "r")
            elif convention == "SimpleHeadphoneIR":
                return sofa.SOFASimpleHeadphoneIR(file.ncfile.filename, "r")
            elif convention == "SingleRoomDRIR":
                return sofa.SOFASingleRoomDRIR(file.ncfile.filename, "r")
            else:
                raise ValueError(f"unknown SOFA convention {convention}!")

        # load file
        log_str = f'opening file "{os.path.relpath(self._file_name)}"'
        try:
            file_sofa = sofa.SOFAFile(self._file_name, "r")
            file_convention = _load_convention(file_sofa)
        except OSError as e:
            raise ValueError(f"{log_str}\n --> {e.strerror}")

        # check validity
        if not file_sofa.isValid():
            warn_str = "invalid SOFA file."
            logger.warning(warn_str) if logger else print(warn_str, file=sys.stderr)
        elif not file_convention.isValid():
            warn_str = (
                f"invalid SOFA file according to "
                f'`{file_sofa.getGlobalAttributeValue("SOFAConventions")}` convention.'
            )
            logger.warning(warn_str) if logger else print(warn_str, file=sys.stderr)

        # generate file info
        log_str = (
            f"{log_str}\n --> samplerate: {file_convention.getSamplingRate()[0]:.0f} Hz"
            f', receivers: {file_convention.ncfile.file.dimensions["R"].size}'
            f', emitters: {file_convention.ncfile.file.dimensions["E"].size}'
            f', measurements: {file_convention.ncfile.file.dimensions["M"].size}'
            f', samples: {file_convention.ncfile.file.dimensions["N"].size}'
            f", format: {file_convention.getDataIR().dtype}"
            f'\n --> convention: {file_convention.getGlobalAttributeValue("SOFAConventions")}'
            f', version: {file_convention.getGlobalAttributeValue("SOFAConventionsVersion")}'
        )
        try:
            log_str = (
                f"{log_str}, "
                f'listener: {file_convention.getGlobalAttributeValue("ListenerDescription")}'
            )
        except sofa.SOFAError:
            pass
        try:
            log_str = f'{log_str}, author: {file_convention.getGlobalAttributeValue("Author")}'
        except sofa.SOFAError:
            pass
        logger.info(log_str) if logger else print(log_str)

    def _load(self, dtype):
        """
        Load a filter set containing impulse responses for provided spherical grid positions.

        Parameters
        ----------
        dtype : str or numpy.dtype or type
            numpy data type the contents should be stored in

        Raises
        ------
        TypeError
            in case unknown type for filter name is given
        ValueError
            in case source / receiver grid given in units not according to the SOFA convention
        ValueError
            in case impulse response data is incomplete
        """

        def _check_irs(irs):
            if isinstance(irs, np.ma.MaskedArray):
                # check that all values exist
                if np.ma.count_masked(irs):
                    raise ValueError(f"incomplete IR data at positions {irs.mask}.")
                # transform into regular `numpy.ndarray`
                irs = irs.filled(0)
            return irs.astype(dtype)  # `astype()` makes copy

        if not isinstance(self._file_name, str):
            raise TypeError(f'unknown parameter type "{type(self._file_name)}".')

        # load file
        file = sofa.SOFAFile(self._file_name, "r")

        # get IRs and adjust data type
        self._irs_td = _check_irs(file.getDataIR())

        # save needed attributes
        self._fs = int(file.getSamplingRate()[0])

        # get grid
        if self._is_hrir:
            # transform grid into azimuth, colatitude, radius in radians
            grid_acr_rad = sfa.utils.SOFA_grid2acr(
                grid_values=file.getSourcePositionValues(),
                grid_info=file.getSourcePositionInfo(),
            )

            # store spherical degree with elevation !!
            self._points_azim_deg = np.rad2deg(grid_acr_rad[0])
            self._points_elev_deg = 90 - np.rad2deg(grid_acr_rad[1])

        else:
            # transform grid into azimuth, colatitude, radius in radians
            grid_acr_rad = sfa.utils.SOFA_grid2acr(
                grid_values=file.getReceiverPositionValues()[:, :, 0],
                grid_info=file.getReceiverPositionInfo(),
            )

            # assume rigid sphere and omnidirectional transducers according to SOFA 1.0, AES69-2015
            self._arir_config = sfa.io.ArrayConfiguration(
                array_radius=grid_acr_rad[2].mean(),
                array_type="rigid",
                transducer_type="omni",
            )

        # store spherical radians with colatitude !!
        self._irs_grid = sfa.io.SphericalGrid(
            azimuth=grid_acr_rad[0], colatitude=grid_acr_rad[1], radius=grid_acr_rad[2]
        )
