import re
import sys
from enum import Enum

import numpy as np
import sound_field_analysis as sfa

from . import tools


class Compensation(object):
    """
    Flexible abstract structure providing functionality to realize different compensation methods
    helpful when processing microphone arrays in the SH domain.
    """

    _TYPES_STR_SEP = r",.;:|/\-+"
    """Characters to separate compensation techniques when concatenated in a string."""
    _TYPES_REQUESTED = []
    """Global list of `Compensation.Type` to store requested compensation techniques,
    filled automatically. """
    _TYPES_APPLIED = []
    """Global list of `Compensation.Type` to store already applied compensation techniques,
    filled automatically. """
    _NFFT_USED = None
    """Global int to store "used" filter taps by already compensation techniques."""
    _IS_PLOT = True
    """Global bool to store if plots of the compensation techniques should be generated and
    exported. """

    _SHF_NFFT = [40, 256]
    """Spherical Head Filter recommended minimum and maximum length in time domain before zero
    padding. This value was chosen based on informal testing. Lower limit leading to a sufficient
    transfer function for e.g. Eigenmike rendering at order 1. Upper limit leading to a
    sufficient transfer function for all rendering conditions with little delay. """

    _MRF_NFFT = [90, -1]
    """Modal Radial Filter recommended minimum and maximum length in time domain before zero
    padding. This value was chosen based on informal testing. Lower limit leading to a sufficient
    transfer function for e.g. Eigenmike rendering at order 4. Upper limit is left open to use
    the remaining amount of available filter length. """

    # noinspection SpellCheckingInspection,PyUnusedName
    class Type(Enum):
        """
        Enumeration data type used to get an identification of SH processing compensation
        techniques. It's attributes (with strings in long form provided for nicer printing
        output) are used as system-wide unique (with alias) constant identifiers.
        """

        SPF = "SUBSONIC_PRE_FILTER"
        SUBSONIC_PRE_FILTER = SPF
        """Subsonic pre-filter by an FIR highpass filter meant to be applied to the array signals
        before spatial decomposition of the sound field.

        This method should not be applied in the current state, since it leads to time aliasing
        due to the block wise processing. If anything, an extended highpass filter with many taps
        should be introduced as a separate `JackRenderer` utilizing partitioned convolution. """

        MRF = "MODAL_RADIAL_FILTER"
        MODAL_RADIAL_FILTER = MRF
        """
        Modal radial filter depending on the microphone array radius and body with adjustable soft
        clipping for frequency compensation of scattering at the microphone body according to [1].
        Radial filters need to be applied in every case of spherical array processing.
        Generation could be improved by [2]?

        [1] Bernschütz, B., Pöschmann, C., Spors, S., and Weinzierl, S. (2011). “Soft-Limiting der
            modalen Amplitudenverstärkung bei sphärischen Mikrofonarrays im Plane Wave
            Decomposition Verfahren,” Fortschritte der Akust. -- DAGA 2011, Deutsche Gesellschaft
            für Akustik, Düsseldorf, Germany, 661–662.
        [2] Zotter, F. (2018). “A Linear-Phase Filter-Bank Approach to Process Rigid Spherical
            Microphone Array Recordings,” IcETRAN, IEEE, Palić, Serbia, 1-8.
        """

        # MRF_MR = 'MODAL_RADIAL_FILTER_MAX_RE'
        # """
        # Modal radial filter depending on the microphone array radius and body for frequency
        # compensation of scattering at the microphone body according to [1].
        # Radial filters need to be applied in every case of spherical array processing.
        # This method is not implemented yet!
        #
        # [1] Zotter, F. (2018). “A Linear-Phase Filter-Bank Approach to Process Rigid Spherical
        #     Microphone Array Recordings,” IcETRAN, IEEE, Palić, Serbia, 1-8.
        # """
        # MODAL_RADIAL_FILTER_MAX_RE = MRF_MR

        SHF = "SPHERICAL_HEAD_FILTER"
        SPHERICAL_HEAD_FILTER = SHF
        """
        Global filter depending on the microphone array radius for frequency compensation and
        azimuth shift of errors due to spherical harmonics order truncation of the HRIR according
        to [1].
        This method should be applied when not using any other order truncation mitigation or HRIR
        pre-processing methods.

        [1] Ben-Hur, Z., Brinkmann, F., Sheaffer, J., Weinzierl, S., and Rafaely, B. (2017).
            “Spectral equalization in binaural signals represented by order-truncated spherical
            harmonics,” J. Acoust. Soc. Am., 141, 4087–4096. doi:10.1121/1.4983652
        """

        SHT = "SPHERICAL_HARMONICS_TAPERING"
        SPHERICAL_HARMONICS_TAPERING = SHT
        """
        Order varying SH weights depending on the used rendering order for compensation of errors
        due to spherical harmonics order truncation of the HRIR according to [1].
        This method should be combined with a slightly adjusted spherical head filter (see
        reference implementation).

        [1] Hold, C., Gamper, H., Pulkki, V., Raghuvanshi, N., and Tashev, I. J. (2019). “Improving
            Binaural Ambisonics Decoding by Spherical Harmonics Domain Tapering and Coloration
            Compensation,” Int. Conf. Acoust. Speech Signal Process., IEEE, Brighton, UK, 261–265.
            doi:10.1109/ICASSP.2019.8683751
        """

        SDS = "SECTORIAL_DEGREE_SELECTION"
        SECTORIAL_DEGREE_SELECTION = SDS
        """
        Weighting (selection) of all (2*N)+1 sectorial SH modes, e.g. (0,0), (1,-1), (1,+1), (2,-2),
        (2,+2), (3,-3), (3,+3). This means all non-sectorial modes (not in the outer diagonal of
        the SH triangle) in the decomposed HRIR will be set to 0.
        This method is experimental and not published. This method should only be applied in
        horizontal synthesis (only considering azimuthal head rotations).
        """

        EDS = "EQUATORIAL_DEGREE_SELECTION"
        EQUATORIAL_DEGREE_SELECTION = EDS
        """
        Weighting (selection) of all sum(range(N+2)) (partial sum of N+1) SH modes containing
        information on the equator, e.g. (0,0), (1,-1), (1,+1), (2,-2), (2,0), (2,+2), (3,-3),
        (3,-1), (3,+1), (3,+3). This means all non-horizontal modes (not in the outer diagonal plus
        every second mode of the SH triangle) in the decomposed HRIR will be set to 0.
        This method is experimental and not published. This method should only be applied in
        horizontal synthesis (only considering azimuthal head rotations).
        """

        AMF = "ALIASING_MITIGATION_FILTER"
        ALIASING_MITIGATION_FILTER = AMF
        """
        Global filter depending on the microphone array radius and used rendering order for
        frequency compensation of errors due to spatial aliasing when capturing and transforming
        the microphone signals according to [1].
        This method can be combined with all other compensation methods.
        This method is not implemented yet!

        [1] Lübeck, T. (unpublished). Master thesis, Technische Hochschule Cologne, Germany.
        """

        MLS = "MAGNITUDE_LEAST_SQUARES"
        MAGNITUDE_LEAST_SQUARES = MLS
        """
        HRIR pre-processing depending on the used rendering order for minimizing error due to
        spherical harmonics order truncation according to [1].
        This method should NOT be combined with other truncation mitigation methods like the
        spherical head filter.
        This method is not implemented yet!

        [1] Schörkhuber, C., Zaunschirm, M., and Höldrich, R. (2018). “Binaural Rendering of
            Ambisonic Signals via Magnitude Least Squares,” Fortschritte der Akust. -- DAGA 2018,
            Deutsche Gesellschaft für Akustik, Munich, Germany, 339–342.
        """

    @staticmethod
    def reset_config(is_plot):
        """
        Reset parameters storing requested and already applied compensation methods. This needs
        to be executed in case compensation techniques should be re-applied i.e., in case
        rendering conditions are changed during runtime.

        Parameters
        ----------
        is_plot : bool
            new state if plots of generated compensation filters should be generated and exported
        """
        Compensation._IS_PLOT = is_plot
        Compensation._TYPES_REQUESTED = []
        Compensation._TYPES_APPLIED = []
        Compensation._NFFT_USED = None

    # noinspection PyProtectedMember
    @staticmethod
    def generate_by_type(
        compensation_types,
        filter_set,
        arir_config=None,
        amp_limit_db=None,
        fc=None,
        nfft=None,
        nfft_padded=None,
        logger=None,
    ):
        """
        Generate superposition of all compensation filters based on provided parameters. Function
        can be called with a list of methods, which will be applied by recursive execution of
        this function.

        Parameters
        ----------
        compensation_types : list of str or list of Compensation.Type or str or Compensation.Type
            type of spherical harmonics compensation for order truncation or aliasing
        filter_set : FilterSet
            FIR filter set that compensation should be applied to
        arir_config : sfa.io.ArrayConfiguration, optional
            recording / measurement microphone array configuration
        amp_limit_db : int, optional
            compensation filter maximum amplification limit in dB
        fc : float, optional
            compensation filter cutoff / critical frequency
        nfft : int, optional
            compensation filter target length
        nfft_padded : int, optional
            compensation filter length after applying zero padding if desired
        logger : logging.Logger, optional
            instance to provide identical logging behaviour as the parent process

        Returns
        -------
        numpy.ndarray
            one-sided complex frequency spectra of all compensation filters

        Raises
        ------
        RuntimeError
            in case frequency domain blocks have not been calculated yet
        ValueError
            in case the adjusted filter length was not within the individually specified limits,
            see `Compensation`
        """

        def _split_types(_types):
            if isinstance(_types, list):
                # if type is list of multiple types or strings
                return [_split_types(t) for t in _types]
            elif isinstance(_types, str):
                if any(sep in _types for sep in Compensation._TYPES_STR_SEP):
                    # if type is string list of multiple types
                    return [
                        _split_types(t)
                        for t in re.split(f"[{Compensation._TYPES_STR_SEP}]+", _types)
                    ]
                else:
                    # if type is string of singular type
                    return tools.transform_into_type(_types, Compensation.Type)
            elif isinstance(_types, Compensation.Type) or _types is None:
                # if type is instance of singular type
                return _types
            else:
                raise ValueError(
                    f"unknown parameter type `{type(_types)}`, see `Compensation.Type` for "
                    f"reference! "
                )

        def _generate_by_type(_type):
            return Compensation.generate_by_type(
                _type,
                filter_set=filter_set,
                arir_config=arir_config,
                amp_limit_db=amp_limit_db,
                nfft=nfft,
                nfft_padded=nfft_padded,
                logger=logger,
            )

        def _adjust_nfft(given, lower_limit, upper_limit):
            # pick standard length depending on filter would be less than half of taps left
            if given / 2 >= upper_limit:
                # if given, pick upper boundary to leave more taps to other filters
                adjusted = upper_limit if upper_limit > 0 else given
            else:
                # pick lower boundary as minimum applicable
                adjusted = lower_limit
            if adjusted < 1 or adjusted < lower_limit:
                raise ValueError(
                    f"Compensation filter target length of {adjusted} samples is too low for type "
                    f'"{_type.value}".'
                )
            return adjusted

        if Compensation._NFFT_USED is None:
            Compensation._NFFT_USED = filter_set._irs_orig_shape[-1]

        is_nfft_given = nfft is not None and nfft > 0
        if not nfft_padded:
            if is_nfft_given:
                # make padded length identical to target length
                nfft_padded = nfft
            else:
                # make padded length identical to given filter
                nfft_padded = filter_set._dirac_td.shape[-1]

        # allocate size of one-sided spectrum in target length
        comps_nm = np.ones(
            nfft_padded // 2 + 1, dtype=filter_set._dirac_blocks_fd.dtype
        )

        # add all compensations to requested list
        if not Compensation._TYPES_REQUESTED:
            compensation_types = _split_types(compensation_types)
            if not isinstance(compensation_types, list):
                compensation_types = [compensation_types]
            for compensation_type in compensation_types:
                if compensation_type not in Compensation._TYPES_REQUESTED:
                    Compensation._TYPES_REQUESTED.append(compensation_type)

        if isinstance(compensation_types, list):
            # make recursive call on types individually
            for compensation_type in compensation_types:
                comps_nm = comps_nm * _generate_by_type(compensation_type)
                # *= does not work because of broadcasting shape error
            return comps_nm

        else:
            # otherwise generate individual type
            _type = tools.transform_into_type(compensation_types, Compensation.Type)
            if _type is None:
                return comps_nm
            if filter_set._irs_blocks_nm is None:
                raise RuntimeError(
                    "filter blocks in spherical harmonics domain have not been calculated yet."
                )

            # make sure each compensation is only applied once
            if _type in Compensation._TYPES_APPLIED:
                log_str = (
                    f'skipping compensation type "{_type.value}" since it was already '
                    f"applied."
                )
                logger.warning(log_str) if logger else print(
                    f"[WARNING]  {log_str}", file=sys.stderr
                )
                return comps_nm

            if _type in [
                Compensation.Type.SHT,
                Compensation.Type.SDS,
                Compensation.Type.EDS,
            ]:
                # filter length is singular weighting factor
                nfft = 1
            else:
                # calculate new individual filter length
                if not is_nfft_given:
                    # pick rest of available taps
                    nfft = filter_set._dirac_td.shape[-1] - Compensation._NFFT_USED + 1
                    # adjust length based on individually set limitations
                    if _type is Compensation.Type.SHF:
                        nfft = _adjust_nfft(nfft, *Compensation._SHF_NFFT)
                    elif _type is Compensation.Type.MRF:
                        nfft = _adjust_nfft(nfft, *Compensation._MRF_NFFT)

                # make target length even
                if nfft % 2:
                    nfft -= 1

            # mark compensation as applied and remember used length
            Compensation._TYPES_APPLIED.append(_type)
            Compensation._NFFT_USED += nfft - 1  # convolution result length = N + M - 1
            # print(
            #     f"type={_type.value}, nfft={nfft}, nfft_used={Compensation._NFFT_USED}"
            # )

            # calculate compensation
            return Compensation._generate_fd_by_type(
                _type,
                sh_max_order=filter_set._sh_max_order,
                nfft=nfft,
                nfft_padded=nfft_padded,
                fs=filter_set._fs,
                arir_config=arir_config,
                amp_limit_db=amp_limit_db,
                fc=fc,
                dtype=filter_set._irs_blocks_nm.dtype,
                logger=logger,
            )

    @staticmethod
    def _generate_fd_by_type(
        _type,
        sh_max_order,
        nfft,
        nfft_padded,
        fs,
        arir_config,
        amp_limit_db,
        fc,
        dtype,
        logger,
    ):
        """
        Generate compensation filter based on individual implementations and apply zero padding if
        desired.

        Parameters
        ----------
        _type : Compensation.Type or str
            type of spherical harmonics compensation for order truncation or aliasing
        sh_max_order : int or None
            maximum spherical harmonics order
        nfft : int
            compensation filter target length
        nfft_padded : int or None
            compensation filter length after applying zero padding if desired
        fs : int
            compensation filter sampling frequency
        arir_config : sfa.io.ArrayConfiguration or None
            recording / measurement microphone array configuration
        amp_limit_db : int or None
            compensation filter maximum amplification limit in dB
        fc : float or None
            compensation filter cutoff / critical frequency
        dtype : str or numpy.dtype or type
            compensation filter numpy data type
        logger : logging.Logger or None
            instance to provide identical logging behaviour as the parent process

        Returns
        -------
        numpy.ndarray
            one-sided complex frequency spectra of the compensation filter
        """
        _type = tools.transform_into_type(_type, Compensation.Type)
        log_str = f'applying "{_type.value}" compensation ...'
        logger.info(log_str) if logger else print(log_str)

        # if amp_limit_db and _type != Compensation.Type.MRF:
        #     log_str = (
        #         f'ignoring amplitude limitation for compensation type "{_type.value}".'
        #     )
        #     logger.debug(log_str) if logger else print(log_str)
        # if fc and _type != Compensation.Type.SPF:
        #     log_str = (
        #         f'ignoring frequency parameter for compensation type "{_type.value}".'
        #     )
        #     logger.debug(log_str) if logger else print(log_str)

        # generate compensation based on individual implementation
        if _type == Compensation.Type.SPF:
            comp_nm = Compensation._generate_fd_spf(
                nfft=nfft, fs=fs, fc=fc, iir_order=4, dtype=dtype, logger=logger
            )
        elif _type == Compensation.Type.MRF:
            comp_nm = Compensation._generate_fd_mrf(
                sh_max_order=sh_max_order,
                nfft=nfft,
                fs=fs,
                dtype=dtype,
                arir_config=arir_config,
                amp_limit_db=amp_limit_db,
                logger=logger,
            )
        elif _type == Compensation.Type.SHF:
            comp_nm = Compensation._generate_fd_shf(
                sh_max_order=sh_max_order,
                nfft=nfft,
                fs=fs,
                radius=arir_config.array_radius,
                dtype=dtype,
                logger=logger,
                is_tapering=Compensation.Type.SHT in Compensation._TYPES_REQUESTED,
            )
        elif _type == Compensation.Type.SHT:
            comp_nm = Compensation._generate_fd_sht(
                sh_max_order=sh_max_order, dtype=dtype, logger=logger
            )
        elif _type == Compensation.Type.SDS:
            comp_nm = Compensation._generate_fd_sds(
                sh_max_order=sh_max_order,
                dtype=dtype,
                logger=logger,
            )
        elif _type == Compensation.Type.EDS:
            comp_nm = Compensation._generate_fd_eds(
                sh_max_order=sh_max_order,
                dtype=dtype,
                logger=logger,
            )
        elif _type == Compensation.Type.AMF:
            comp_nm = Compensation._generate_fd_amf(
                sh_max_order=sh_max_order,
                nfft=nfft,
                fs=fs,
                arir_config=arir_config,
                dtype=dtype,
                logger=logger,
            )
        elif _type == Compensation.Type.MLS:
            comp_nm = Compensation._generate_fd_mls(
                sh_max_order=sh_max_order,
                nfft=nfft,
                fs=fs,
                arir_config=arir_config,
                dtype=dtype,
                logger=logger,
            )
        else:
            raise ValueError(
                f'unknown compensation type "{_type}", see `Compensation.Type` for reference!'
            )

        if comp_nm.shape[-1] == 1:
            # repeat to two frequency bins to prevent exception inverse Fourier transform
            comp_nm = np.repeat(comp_nm, 2, axis=-1)

        # apply zero padding to desired length
        return sfa.utils.zero_pad_fd(data_fd=comp_nm, target_length_td=nfft_padded)

    @staticmethod
    def _generate_fd_spf(nfft, fs, fc, iir_order, dtype, logger):
        """
        Generate subsonic pre-filter, see `Compensation.Type` for reference.

        Parameters
        ----------
        nfft : int
            compensation filter length
        fs : int
            compensation filter sampling frequency
        fc : float
            compensation highpass filter cutoff frequency
        iir_order : int
            compensation highpass filter equivalent IIR order
        dtype : str or numpy.dtype or type
            numpy data type of generated array
        logger : logging.Logger
            instance to provide identical logging behaviour as the parent process

        Returns
        -------
        numpy.ndarray
            one-sided complex frequency spectra of the compensation filter
        """
        comp_fd = tools.generate_iir_filter_fd(
            type_str="highpass",
            length_td=nfft,
            fs=fs,
            fc=fc,
            iir_order=iir_order,
            is_lr=False,
            is_linear_phase=False,
            is_apply_window=True,
        )

        comp_fd = comp_fd.astype(dtype)  # adjust dtype
        if Compensation._IS_PLOT:
            Compensation._plot_ir_and_tf(
                comp_nm=comp_fd,
                _type=Compensation.Type.SPF,
                sh_max_order=None,
                nfft=nfft,
                fs=fs,
                logger=logger,
            )
        return comp_fd

    @staticmethod
    def _generate_fd_mrf(
        sh_max_order, nfft, fs, arir_config, amp_limit_db, dtype, logger
    ):
        """
        Generate modal radial filter, see `Compensation.Type` for reference.

        Parameters
        ----------
        sh_max_order : int
            maximum spherical harmonics order
        nfft : int
            compensation filter length
        fs : int
            compensation filter sampling frequency
        arir_config : sfa.io.ArrayConfiguration
            recording / measurement microphone array configuration
        amp_limit_db : int
             maximum modal amplification limit in dB
        dtype : str or numpy.dtype or type
            numpy data type of generated array
        logger : logging.Logger
            instance to provide identical logging behaviour as the parent process

        Returns
        -------
        numpy.ndarray
            weights of the compensation function which can be used as one-sided complex frequency
            spectra or real time domain factors
        """
        # calculate for all SH orders
        comp_nm = sfa.gen.radial_filter_fullspec(
            max_order=sh_max_order,
            NFFT=nfft,
            fs=fs,
            array_configuration=arir_config,
            amp_maxdB=amp_limit_db,
        )

        # apply improvement (remove DC offset, make causal and windowing)
        comp_nm, _, comp_delay = sfa.process.rfi(comp_nm, kernelSize=nfft)
        comp_nm = comp_nm.astype(dtype)  # adjust dtype
        log_str = f"introduced compensation causal shift delay of {1000 * comp_delay / fs:.1f} ms."
        logger.info(log_str) if logger else print(log_str)

        if Compensation._IS_PLOT:
            # plot for all SH orders
            Compensation._plot_ir_and_tf(
                comp_nm=comp_nm,
                _type=Compensation.Type.MRF,
                sh_max_order=sh_max_order,
                nfft=nfft,
                fs=fs,
                mrf_params=(
                    amp_limit_db,
                    arir_config.array_type,
                    arir_config.transducer_type,
                ),
                logger=logger,
            )

        # repeat to match number of SH degrees
        comp_nm = np.repeat(
            comp_nm[:, np.newaxis, :], range(1, sh_max_order * 2 + 2, 2), axis=0
        )

        # combine SH coefficients
        sh_m = sfa.sph.mnArrays(sh_max_order)[0].astype(np.int16)
        sh_m_power = (
            arir_config.array_radius.dtype.type(-1.0) ** sh_m[:, np.newaxis, np.newaxis]
        )  # inherit dtype

        return comp_nm * sh_m_power

    @staticmethod
    def _generate_fd_shf(sh_max_order, nfft, fs, radius, is_tapering, dtype, logger):
        """
        Generate global spherical head compensation filter, see `Compensation.Type` for reference.

        Parameters
        ----------
        sh_max_order : int
            maximum spherical harmonics order
        nfft : int
            compensation filter length
        fs : int
            compensation filter sampling frequency
        radius : float
            recording / measurement microphone array radius in meter
        is_tapering : bool
            if Spherical Harmonics Tapering should be or has already been applied as well
        dtype : str or numpy.dtype or type
            numpy data type of generated array
        logger : logging.Logger
            instance to provide identical logging behaviour as the parent process

        Returns
        -------
        numpy.ndarray
            one-sided complex frequency spectra of the compensation filter
        """
        # ignore invalid value FloatingPointError (only encountered on Windows)
        with np.errstate(invalid="ignore", under="ignore"):
            # calculate globally
            comp_nm = sfa.gen.spherical_head_filter_spec(
                max_order=sh_max_order,
                NFFT=nfft,
                fs=fs,
                radius=radius,
                is_tapering=is_tapering,
            )
        comp_nm = comp_nm.astype(dtype)  # adjust dtype

        # make causal
        comp_delay = nfft // 2
        comp_nm *= sfa.gen.delay_fd(
            target_length_fd=comp_nm.shape[-1], delay_samples=comp_delay
        )
        log_str = (
            f"introduced compensation filter causal shift delay of "
            f"{1000 * comp_delay / fs:.1f} ms."
        )
        logger.info(log_str) if logger else print(log_str)

        if Compensation._IS_PLOT:
            # plot globally
            Compensation._plot_ir_and_tf(
                comp_nm=comp_nm,
                _type=Compensation.Type.SHF,
                sh_max_order=sh_max_order,
                nfft=nfft,
                fs=fs,
                logger=logger,
            )

        return comp_nm

    @staticmethod
    def _generate_fd_sht(sh_max_order, dtype, logger):
        """
        Generate order varying SH tapering weights, see `Compensation.Type` for reference.

        Parameters
        ----------
        sh_max_order : int
            maximum spherical harmonics order
        dtype : str or numpy.dtype or type
            numpy data type of generated array
        logger : logging.Logger
            instance to provide identical logging behaviour as the parent process

        Returns
        -------
        numpy.ndarray
             weights of the compensation function which can be used as one-sided complex frequency
             spectra or real time domain factors
        """
        # calculate for all SH orders
        weights_nm = sfa.gen.tapering_window(max_order=sh_max_order).astype(dtype)

        # repeat to match number of SH degrees
        weights_nm = np.repeat(
            weights_nm[:, np.newaxis, np.newaxis],
            range(1, sh_max_order * 2 + 2, 2),
            axis=0,
        )

        if Compensation._IS_PLOT:
            # plot for all SH degrees
            Compensation._plot_nm_weights(
                weights_nm=weights_nm,
                _type=Compensation.Type.SHT,
                sh_max_order=sh_max_order,
                logger=logger,
            )

        return weights_nm

    @staticmethod
    def _generate_fd_sds(sh_max_order, dtype, logger):
        """
        Generate SH weights selecting only sectorial modes, see `Compensation.Type` for reference.

        Parameters
        ----------
        sh_max_order : int
            maximum spherical harmonics order
        dtype : str or numpy.dtype or type
            numpy data type of generated array
        logger : logging.Logger
            instance to provide identical logging behaviour as the parent process

        Returns
        -------
        numpy.ndarray
            weights of the selection function which can be used as one-sided complex frequency
            spectra or real time domain factors
        """

        def _get_non_sectorial_ids():
            ids = []
            for n in range(1, sh_max_order + 1, 1):
                ids.extend(list(range(n**2 + 1, (n + 1) ** 2 - 1, 1)))
            return ids

        # def _get_sectorial_ids():
        #     ids = []
        #     for n in range(sh_max_order + 1):
        #         ids.append(n**2)
        #         if n > 0:
        #             ids.append((n + 1) ** 2 - 1)
        #     return ids

        # generate weights according to the number of SH degrees
        weights_nm = np.ones(((sh_max_order + 1) ** 2, 1, 1), dtype=dtype)

        # apply zeros at respective SH degrees
        weights_nm[_get_non_sectorial_ids(), :, :] = 0

        if Compensation._IS_PLOT:
            # plot for all SH degrees
            Compensation._plot_nm_weights(
                weights_nm=weights_nm,
                _type=Compensation.Type.SDS,
                sh_max_order=sh_max_order,
                logger=logger,
            )

        return weights_nm

    @staticmethod
    def _generate_fd_eds(sh_max_order, dtype, logger):
        """
        Generate SH weights selecting only equatorial modes, see `Compensation.Type` for reference.

        Parameters
        ----------
        sh_max_order : int
            maximum spherical harmonics order
        dtype : str or numpy.dtype or type
            numpy data type of generated array
        logger : logging.Logger
            instance to provide identical logging behaviour as the parent process

        Returns
        -------
        numpy.ndarray
            weights of the selection function which can be used as one-sided complex frequency
            spectra or real time domain factors
        """

        def _get_non_equatorial_ids():
            ids = []
            for n in range(1, sh_max_order + 1, 1):
                ids.extend(list(range(n**2 + 1, (n + 1) ** 2, 2)))
            return ids

        # def _get_equatorial_ids():
        #     ids = []
        #     for n in range(sh_max_order + 1):
        #         ids.extend(list(range(n**2, (n + 1) ** 2, 2)))
        #     return ids

        # generate weights according to the number of SH degrees
        weights_nm = np.ones(((sh_max_order + 1) ** 2, 1, 1), dtype=dtype)

        # apply zeros at respective SH degrees
        weights_nm[_get_non_equatorial_ids(), :, :] = 0

        if Compensation._IS_PLOT:
            # plot for all SH degrees
            Compensation._plot_nm_weights(
                weights_nm=weights_nm,
                _type=Compensation.Type.EDS,
                sh_max_order=sh_max_order,
                logger=logger,
            )

        return weights_nm

    @staticmethod
    def _generate_fd_amf(sh_max_order, nfft, fs, arir_config, dtype, logger):
        # TODO: implement aliasing mitigation filter
        raise NotImplementedError(
            f'chosen compensation type "{Compensation.Type.AMF}" is not implemented yet.'
        )

    @staticmethod
    def _generate_fd_mls(sh_max_order, nfft, fs, arir_config, dtype, logger):
        # TODO: implement MagLS pre-processing method
        raise NotImplementedError(
            f'chosen compensation type "{Compensation.Type.MLS}" is not implemented yet.'
        )

    @staticmethod
    def _plot_ir_and_tf(
        comp_nm, _type, sh_max_order, nfft, fs, mrf_params=None, logger=None
    ):
        # build name
        name = f"{logger.name if logger else _type.__module__}_{_type.name}_{nfft}"
        if sh_max_order is not None:
            name = f"{name}_sh{sh_max_order}"
        if mrf_params is not None:
            name = (
                f"{name}_{mrf_params[0]}db_{'_'.join(str(e) for e in mrf_params[1:])}"
            )

        # plot filters TD and FD
        tools.export_plot(
            figure=tools.plot_ir_and_tf(data_td_or_fd=comp_nm, fs=fs),
            name=name,
            logger=logger,
        )

    @staticmethod
    def _plot_nm_weights(weights_nm, _type, sh_max_order, logger=None):
        # build name
        name = f"{logger.name if logger else _type.__module__}"
        name = f"{name}_{_type.name}_sh{sh_max_order}"

        # plot weights
        tools.export_plot(
            figure=tools.plot_nm_rms(data_nm_fd=weights_nm),
            name=name,
            logger=logger,
        )
