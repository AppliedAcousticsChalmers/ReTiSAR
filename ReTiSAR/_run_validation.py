import inspect
import os

import matplotlib.pyplot as plt
import natsort
import numpy as np
import pandas as pd
from scipy.optimize import fmin

from . import config, FilterSet, process_logger, tools


# def main_snr():
#     """
#     Function containing the entire validation procedure including defining configuration,
#     running the validation against provided reference impulse response set as well as exporting
#     and visualizing the results.
#
#     Returns
#     -------
#     logging.Logger
#         logger instance
#     """
#     import soundfile
#
#     # ----------------------------------------
#     # # BEGIN: DEFINE VALIDATION CONFIGURATION
#     # ----------------------------------------
#     _NOISE_IN_FILE = "log/SNR_4096_noise_in.wav"
#     _NOISE_OUT_FILE = "log/SNR_4096_noise_out.wav"
#     _SIGNAL_IN_FILE = "log/SNR_4096_signal_in.wav"
#     _SIGNAL_OUT_FILE = "log/SNR_4096_signal_out.wav"
#     # ----------------------------------------
#     # END:     DEFINE VALIDATION CONFIGURATION
#     # ----------------------------------------
#
#     _NOISE_IN_FILE = tools.get_absolute_from_relative_package_path(_NOISE_IN_FILE)
#     _NOISE_OUT_FILE = tools.get_absolute_from_relative_package_path(_NOISE_OUT_FILE)
#     _SIGNAL_IN_FILE = tools.get_absolute_from_relative_package_path(_SIGNAL_IN_FILE)
#     _SIGNAL_OUT_FILE = tools.get_absolute_from_relative_package_path(_SIGNAL_OUT_FILE)
#
#     # prepare logger
#     name = f'{__package__}{os.path.basename(__file__).strip(".py")}'
#     logger = process_logger.setup(name)
#
#     # read recorded files
#     noise_in_td, fs = soundfile.read(_NOISE_IN_FILE, dtype=np.float32)
#     noise_out_td, fs = soundfile.read(_NOISE_OUT_FILE, dtype=np.float32)
#     signal_in_td, fs = soundfile.read(_SIGNAL_IN_FILE, dtype=np.float32)
#     signal_out_td, fs = soundfile.read(_SIGNAL_OUT_FILE, dtype=np.float32)
#
#     # truncate recordings to shortest length
#     length = min(
#         [
#             noise_in_td.shape[0],
#             noise_out_td.shape[0],
#             signal_in_td.shape[0],
#             signal_out_td.shape[0],
#         ]
#     )
#     noise_in_td = noise_in_td[:length, 0].T
#     noise_out_td = noise_out_td[:length, 0].T
#     signal_in_td = signal_in_td[:length, 0].T
#     signal_out_td = signal_out_td[:length, 0].T
#
#     # calculate differences
#     noise_diff_td = np.fft.irfft(
#         np.abs(np.fft.rfft(noise_out_td, axis=-1))
#         / np.abs(np.fft.rfft(noise_in_td, axis=-1)),
#         axis=-1,
#     )
#     signal_diff_td = np.fft.irfft(
#         np.abs(np.fft.rfft(signal_out_td, axis=-1))
#         / np.abs(np.fft.rfft(signal_in_td, axis=-1)),
#         axis=-1,
#     )
#
#     # generate plots
#     noise_stack_td = np.vstack([noise_in_td, noise_out_td, noise_diff_td])
#     signal_stack_td = np.vstack([signal_in_td, signal_out_td, signal_diff_td])
#     _plot_result(
#         ref_ir=os.path.splitext(os.path.relpath(_NOISE_OUT_FILE))[0],
#         cmp_ir=noise_stack_td,
#         fs=fs,
#     )
#     _plot_result(
#         ref_ir=os.path.splitext(os.path.relpath(_SIGNAL_OUT_FILE))[0],
#         cmp_ir=signal_stack_td,
#         fs=fs,
#     )
#
#     return logger


def main():
    """
    Function containing the entire validation procedure including defining configuration,
    running the validation against provided reference impulse response set as well as exporting
    and visualizing the results.

    Returns
    -------
    logging.Logger
        logger instance
    """
    # ----------------------------------------
    # # BEGIN: DEFINE VALIDATION CONFIGURATION
    # ----------------------------------------
    _CMP_FILES_DIR = "res/research/validation"
    """Path of directory being scanned for comparative impulse response sets."""

    _TRUNC_VOLUME_DB = -100
    # _TRUNC_VOLUME_DB = 0
    """Volume in dB the shorter impulse response set will be truncated to before comparison after
    it reached the last sample over the specified level. """

    _IS_REF_NORM_INDIVIDUALLY = True
    # _IS_REF_NORM_INDIVIDUALLY = False
    """If reference IR channels should be normalized individually for better comparability of RMS
    error. """
    # ----------------------------------------
    # END:     DEFINE VALIDATION CONFIGURATION
    # ----------------------------------------

    # prepare logger
    name = f'{__package__}{os.path.basename(__file__).strip(".py")}'
    logger = process_logger.setup(name)

    # prepare empty data table
    results = pd.DataFrame()

    # check provided comparative IRs azimuth offset
    cmp_azimuth_offset = config.SOURCE_POSITIONS
    if len(cmp_azimuth_offset) > 1:
        logger.error(
            f'given comparative IRs azimuth offset "{cmp_azimuth_offset}" does not contain one '
            f"position tuple."
        )
        return logger
    elif len(cmp_azimuth_offset) == 1 and cmp_azimuth_offset[0][0]:
        cmp_azimuth_offset = cmp_azimuth_offset[0][
            0
        ]  # only keep azimuth of first tuple
        logger.warning(
            f"[INFO]  considering azimuth offset {cmp_azimuth_offset} degree with all comparative "
            f"IRs ..."
        )
    else:
        cmp_azimuth_offset = 0

    # check provided reference IR
    ref_file = tools.get_absolute_from_relative_package_path(config.VALIDATION_MODE)
    if not os.path.isfile(ref_file):
        logger.error(f'given reference IR "{ref_file}" not found.')
        return logger

    # load reference IR set and normalize
    ref_set = FilterSet.create_instance_by_type(ref_file, FilterSet.Type.HRIR_SSR)
    ref_set.load(
        block_length=None,
        is_single_precision=False,
        logger=logger,
        is_prevent_resampling=True,
        is_normalize=True,
        is_normalize_individually=_IS_REF_NORM_INDIVIDUALLY,
    )
    # noinspection PyProtectedMember
    fs = ref_set._fs

    # load and analyze all comparative IR sets
    _CMP_FILES_DIR = tools.get_absolute_from_relative_package_path(_CMP_FILES_DIR)
    try:
        cmp_files = _parse_files_and_azimuth(
            path=_CMP_FILES_DIR,
            file_name_start="rec_Impulse_",
            file_name_end="_target_out.wav",
            azimuth_offset_deg=cmp_azimuth_offset,
            is_reversed=False,
        )
    except ValueError as e:
        logger.error(e)
        return logger
    if not len(cmp_files[0]):
        logger.error(
            f'given comparative IR sets in "{os.path.relpath(_CMP_FILES_DIR)}" not found.'
        )
        return logger

    for cmp_file, cmp_azimuth in zip(*cmp_files):
        print(tools.SEPARATOR)

        # load current comparative IR set
        cmp_file = tools.get_absolute_from_relative_package_path(cmp_file)
        cmp_set = FilterSet.create_instance_by_type(cmp_file, FilterSet.Type.HRIR_SSR)
        try:
            cmp_set.load(
                block_length=None,
                is_single_precision="SP" in cmp_file.upper(),
                logger=logger,
                is_prevent_resampling=True,
                is_normalize=False,
                check_fs=fs,
                is_prevent_logging=True,
            )
        except IndexError:
            logger.info(
                f'... skipping comparative IR set "{os.path.relpath(cmp_file)}".'
            )
            continue

        # only keep IRs
        logger.info(
            f"selecting IRs from head orientation azimuth {cmp_azimuth:.1f} deg, "
            f"elevation {0:.1f} deg."
        )
        ref_ir = ref_set.get_filter_td(azim_deg=-cmp_azimuth, elev_deg=0)
        cmp_ir = cmp_set.get_filter_td()
        if ref_ir.shape[0] != cmp_ir.shape[0]:
            logger.error(
                f"given IRs have different number of channels ({ref_ir.shape[0]} and "
                f"{cmp_ir.shape[0]})."
            )
            return logger

        # truncate shorter filter
        if ref_ir.shape[-1] < cmp_ir.shape[-1]:
            ref_ir = _trunc_ir(
                ir=ref_ir,
                cutoff_db=_TRUNC_VOLUME_DB,
                fs=fs,
                file_name=ref_file,
                logger=logger,
            )
        else:
            cmp_ir = _trunc_ir(
                ir=cmp_ir,
                cutoff_db=_TRUNC_VOLUME_DB,
                fs=fs,
                file_name=cmp_file,
                logger=logger,
            )

        # get time alignment, check for being identical for every channel
        cmp_corr_n = np.zeros(ref_ir.shape[0], dtype=np.int32)
        for ch in range(len(cmp_corr_n)):
            cmp_corr_n[ch] = np.argmax(
                np.correlate(cmp_ir[ch], ref_ir[ch], mode="valid")
            )
        # check match of correlation times of all channels
        if (cmp_corr_n == cmp_corr_n[0]).all():
            cmp_corr_n = cmp_corr_n[0]
        else:
            # arir = sfa.io.read_miro_struct(_ARIR_REF_FILE)
            # _ARIR_RADIUS = float(arir.configuration.scatter_radius)
            # in meters, from 'res/ARIR/CR1_VSA_110RS_L_struct'
            _ARIR_RADIUS = 0.08749999850988388

            # in seconds
            delay = ((cmp_corr_n[0] - cmp_corr_n[1]) / 2) / fs

            # before inverse sine
            angle = delay * tools.SPEED_OF_SOUND / _ARIR_RADIUS

            log_str = (
                f"time alignment is not identical for all channels "
                f"({tools.get_pretty_delay_str(samples=cmp_corr_n, fs=fs)}).\n"
                f" --> e.g. as per ear offset resulting from horizontal head rotation ("
                f"{tools.get_pretty_delay_str(samples=(cmp_corr_n[0] - cmp_corr_n[1]) / 2, fs=fs)}"
                f").\n --> e.g. with {_ARIR_RADIUS * 100:.1f} cm head / array radius"
            )
            if np.abs(angle) > 1:
                logger.error(
                    f"{log_str} such a large offset could not be explained by an azimuth rotation."
                )
            else:
                logger.error(
                    f"{log_str} might be explained by an azimuth rotation offset of approx. "
                    f"{np.rad2deg(np.arcsin(angle)):.1f} deg."
                )

            # # generate plot to visualize different time alignment
            # ir_plot = np.vstack(
            #     [
            #         np.correlate(cmp_ir[ch], ref_ir[ch], mode="valid")[
            #             np.min(cmp_corr_n) - 100 : np.max(cmp_corr_n) + 100
            #         ]
            #         for ch in range(len(cmp_corr_n))
            #     ]
            # )
            # tools.export_plot(
            #     figure=tools.plot_ir_and_tf(ir_plot, fs=fs, is_share_y=True),
            #     name="corr_mismatch",
            #     logger=logger,
            # )
            #
            # for ch in range(len(cmp_corr_n)):
            #     ir_plot = np.vstack(
            #         [
            #             ref_ir[ch, :256],
            #             cmp_ir[ch, cmp_corr_n[ch] : cmp_corr_n[ch] + 256],
            #         ]
            #     )
            #     tools.export_plot(
            #         figure=tools.plot_ir_and_tf(ir_plot, fs=fs, is_share_y=True),
            #         name=f"corr_mismatch_ch{ch}",
            #         logger=logger,
            #     )

            return logger

        # align and truncate to reference IR
        cmp_ir = cmp_ir[:, cmp_corr_n : cmp_corr_n + ref_ir.shape[-1]].copy()
        logger.info(
            f'time aligned "{os.path.relpath(cmp_file)}" to '
            f"{tools.get_pretty_delay_str(samples=cmp_corr_n, fs=fs)}."
        )

        def _calculate_rms_error(scale, ir_scaled, ir):
            return tools.calculate_rms(ir - (scale * ir_scaled))

        # stack channels for equal level alignment
        ref_ir_stack = np.hstack([ch for ch in ref_ir])
        cmp_ir_stack = np.hstack([ch for ch in cmp_ir])
        # perform least RMS fit and level alignment
        cmp_scale = fmin(
            func=_calculate_rms_error,
            x0=1,
            args=(cmp_ir_stack, ref_ir_stack),
            disp=False,
            full_output=True,
        )
        cmp_scale_db = 20 * np.log10(cmp_scale[0][0])
        logger.info(
            f'level aligned "{os.path.relpath(cmp_file)}" by {cmp_scale_db:+.1f} dB.\n'
            f" --> current function value: {cmp_scale[1]:f}, iterations: {cmp_scale[2]:d}, "
            f"function evaluations: {cmp_scale[3]:d}."
        )
        # only keep RMS fit result
        cmp_scale = cmp_scale[0]
        cmp_ir *= cmp_scale

        # calculate difference
        diff_ir = ref_ir - cmp_ir
        diff_rms = tools.calculate_rms(diff_ir, is_level=True)
        logger.info(
            f"calculated RMS difference is "
            f'{np.array2string(diff_rms, precision=1, floatmode="fixed")} dBFS.'
        )
        diff_max = tools.calculate_peak(diff_ir, is_level=True)
        logger.info(
            f"calculated MAX difference is "
            f'{np.array2string(diff_max, precision=2, floatmode="fixed")} dBFS.'
        )

        # plot difference (cut file extension as name)
        name = os.path.splitext(os.path.relpath(cmp_file))[0]
        tools.export_plot(
            figure=_plot_result(ref_ir=ref_ir, cmp_ir=cmp_ir, fs=fs),
            name=name,
            logger=logger,
            file_type="pdf",
        )
        # plot unweighted difference
        diff_tf_w = np.asarray(
            np.abs(np.fft.rfft(diff_ir) / np.fft.rfft(ref_ir)), dtype=np.complex_
        )
        tools.export_plot(
            figure=tools.plot_ir_and_tf(diff_tf_w, fs=fs, is_draw_td=False),
            name=f"{name}_diff_FD_weighted",
            logger=logger,
            file_type="pdf",
        )
        # plot weighted difference
        tools.export_plot(
            figure=tools.plot_ir_and_tf(
                np.abs(diff_ir / ref_ir),
                fs=fs,
                is_etc=True,
                set_td_db_y=100,
                is_draw_fd=False,
            ),
            name=f"{name}_diff_ETC_weighted",
            logger=logger,
            file_type="pdf",
        )

        # save results
        results = results.append(
            _generate_table_entry(
                cmp_file_name=os.path.relpath(cmp_file),
                cmp_azimuth_deg=cmp_azimuth,
                cmp_corr_n_samples=cmp_corr_n,
                cmp_scale_dbfs=np.array2string(
                    cmp_scale_db, precision=1, floatmode="fixed"
                ),
                diff_rms_dbfs=np.array2string(diff_rms, precision=1, floatmode="fixed"),
                diff_max_dbfs=np.array2string(diff_max, precision=2, floatmode="fixed"),
            ),
            ignore_index=True,
        )

    print(tools.SEPARATOR)
    # generate and save HTML file
    # noinspection PyUnboundLocalVariable
    html_title = f'"{os.path.relpath(ref_file)}" reference, {fs:d} Hz, {ref_ir.shape[-1]:d} samples'
    if _TRUNC_VOLUME_DB:
        html_title = f"{html_title} (truncated at {_TRUNC_VOLUME_DB:.1f} dB)"
    else:
        html_title = f"{html_title} (not truncated)"
    html_file = os.path.join(
        _CMP_FILES_DIR,
        f"{os.path.splitext(os.path.basename(ref_file))[0]}{os.path.extsep}html",
    )
    logger.info(f'writing results to "{os.path.relpath(html_file)}" ...')
    tools.export_html(html_file, results.style.render(), title=html_title)

    # end application
    logger.info("... validation ended.")
    return logger


def _parse_files_and_azimuth(
    path, file_name_start, file_name_end, azimuth_offset_deg=0, is_reversed=False
):
    """
    Gather all files according to provided name scheme contained in provided path. Out of the
    individual filenames an azimuth specifying the recorded head rotation is parsed.

    Parameters
    ----------
    path : str
        path to directory that will be analysed
    file_name_start : str
        begin of file names that will be analysed
    file_name_end : str
        end of file names that will be analysed
    azimuth_offset_deg : float, optional
        azimuth offset in degrees that is added to the parsed azimuth from the file name
    is_reversed : bool, optional
        if matched file names should be inversely sorted

    Returns
    -------
    tuple of list of str and list of float
        list of all matched file names and the according azimuths including the provided offset
    """
    files = []
    azimuths_deg = []
    for r, _, f in os.walk(path):
        for file in natsort.natsorted(f, reverse=is_reversed):
            if file.startswith(file_name_start) and file.endswith(file_name_end):
                files.append(os.path.join(r, file))
                # parse azimuth out of comparative IR
                az = [s.strip("deg") for s in file.split("_") if "deg" in s]
                if not len(az) or len(az) > 1:
                    raise ValueError(
                        f'azimuth value in comparative IR set name "{file}" not found.'
                    )
                azimuths_deg.append(float(az[0]) + azimuth_offset_deg)
    # return files[:1], azimuths_deg[:1]
    return files, azimuths_deg


def _trunc_ir(ir, cutoff_db, fs, file_name, logger):
    """
    Truncate impulse response set at the given cutoff level. The new length is calculated by all
    channels having decayed to the provided level relative under global peak.

    Parameters
    ----------
    ir : numpy.ndarray
        loaded impulse response set of size [number of channels; number of samples]
    cutoff_db : float
        truncation level in dB relative under global peak
    fs : int
        loaded impulse response set sampling frequency
    file_name : str
        file name for better logging behaviour
    logger : logging.Logger
        instance to provide identical logging behaviour as the parent process

    Returns
    -------
    numpy.ndarray
        truncated impulse response set of size [number of channels; number of samples]
    """
    # get level under peak as amplitude
    amp_trunc = np.abs(ir).max() * 10 ** (cutoff_db / 20)

    # get first sample over cutout amplitude from the front
    n_trunc = np.argmax(np.abs(ir) > amp_trunc) - 1
    # get first sample over cutout amplitude from the back
    n_trunc_rev = np.argmax(np.flip(np.abs(ir), axis=-1) > amp_trunc) - 1

    # truncate IR if index found
    if n_trunc_rev > 0:
        ir = ir[:, :-n_trunc_rev]
        logger.info(
            f'truncated back of "{os.path.relpath(file_name)}" to {-cutoff_db} dB under peak at '
            f"{tools.get_pretty_delay_str(samples=ir.shape[-1], fs=fs)}."
        )
    if n_trunc > 0:
        ir = ir[:, n_trunc:]
        logger.info(
            f'truncated front of "{os.path.relpath(file_name)}" to {-cutoff_db} dB under peak at '
            f"{tools.get_pretty_delay_str(samples=n_trunc, fs=fs)}."
        )
    elif n_trunc_rev <= 0:
        logger.info(
            f'kept "{os.path.relpath(file_name)}" at '
            f"{tools.get_pretty_delay_str(samples=ir.shape[-1], fs=fs)}."
        )

    return ir


def _plot_result(ref_ir, cmp_ir, fs, compare_frame_samples=128, is_show_fd=False):
    """
    Parameters
    ----------
    ref_ir : numpy.ndarray
        time domain (real) reference data
    cmp_ir : numpy.ndarray
        time domain (real) comparative data
    fs : int
        sampling frequency of data
    compare_frame_samples : int, optional
        number of samples that should be plotted IR around peak (reference is 0th order)
    is_show_fd : bool, optional
        if third column containing frequency domain should be plotted

    Returns
    -------
    matplotlib.figure.Figure
        generated plot
    """

    def _remove_inner_tick_labels(ax, row, col):
        xlabels = ax[row, col].get_xticklabels()
        if row < diff_ir.shape[0] - 1:
            xlabels = []  # remove all elements
        else:
            # if col < _N_COLS - 1:
            #     xlabels[-1] = ''  # remove highest element
            if col > 0:
                xlabels[0] = ""  # remove lowest element
        ax[row, col].set_xticklabels(xlabels)

        ylabels = ax[row, col].get_yticklabels()
        if 0 < col < _N_COLS - 1:
            ylabels = []  # remove all elements
        else:
            if row < diff_ir.shape[0] - 1:
                ylabels[0] = ""  # remove lowest element
            if row > 0:
                ylabels[-1] = ""  # remove highest element
        ax[row, col].set_yticklabels(ylabels)

    from matplotlib.ticker import FuncFormatter

    _N_COLS = 2 + is_show_fd
    _LIM_PEAK_Y = [-1, 1]
    _LIM_ETC_Y = [-140, 0]
    _STEP_ETC_Y = 20
    _STEP_TD_X = 4096
    _FREQS_LABELED = [1, 10, 100, 1000, 10000, 100000]  # labeled frequencies
    _STEP_FD_Y = 5
    _LIM_FD_Y = [-30, 30]

    # calculate difference
    diff_ir = ref_ir - cmp_ir
    x = np.arange(0, diff_ir.shape[-1])
    ref_etc = ref_ir.copy()

    if is_show_fd:
        # calculate spectra
        ref_tf = 20 * np.log10(np.abs(np.fft.rfft(ref_ir)))
        cmp_tf = 20 * np.log10(np.abs(np.fft.rfft(cmp_ir)))
        diff_tf = 20 * np.log10(np.abs(np.fft.rfft(diff_ir)))
        freqs = np.linspace(0, fs / 2, len(diff_tf[1]))
        # _FREQS_LABELED.extend([20, fs / 2])  # add labels at upper and lower frequency limit
        _FREQS_LABELED.append(fs / 2)  # add label at upper frequency limit

    # prevent zeros
    diff_ir[np.nonzero(diff_ir == 0)] = np.nan
    ref_etc[np.nonzero(ref_etc == 0)] = np.nan
    # transform td data into logarithmic scale
    diff_ir = 20 * np.log10(np.abs(diff_ir))
    ref_etc = 20 * np.log10(np.abs(ref_etc))

    fig, axes = plt.subplots(nrows=diff_ir.shape[0], ncols=_N_COLS, squeeze=False)
    for ch in range(diff_ir.shape[0]):
        # select color and plot string
        color = f"C{ch:d}"

        # plot ETC comparison
        axes[ch, 0].plot(x, ref_etc[ch], linewidth=0.5, alpha=0.1, color="black")
        axes[ch, 0].plot(x, diff_ir[ch], linewidth=0.5, color=color)

        if ch == diff_ir.shape[0] - 1:
            axes[ch, 0].set_xlabel("Samples")
        axes[ch, 0].set_ylabel(f"ETC / dBFS")
        axes[ch, 0].tick_params(
            which="both", direction="in", top=True, bottom=True, left=True, right=True
        )
        axes[ch, 0].set_xticks(np.arange(0, np.max(x) + 1, _STEP_TD_X), minor=False)
        axes[ch, 0].grid(True, which="major", axis="x", alpha=0.25)
        axes[ch, 0].set_yticks(
            np.arange(_LIM_ETC_Y[0], _LIM_ETC_Y[1] + _STEP_ETC_Y / 2, _STEP_ETC_Y / 2),
            minor=True,
        )
        axes[ch, 0].set_yticks(
            np.arange(_LIM_ETC_Y[0], _LIM_ETC_Y[1] + _STEP_ETC_Y, _STEP_ETC_Y),
            minor=False,
        )
        axes[ch, 0].grid(True, which="both", axis="y", alpha=0.1)
        axes[ch, 0].set_xlim(0, np.max(x))
        axes[ch, 0].set_ylim(*_LIM_ETC_Y)
        # axes[ch, 0].set_zorder(-1)
        legend_str = [
            "$E_{L,ref}$",
            "$|E_{L,ref}-E_{L,cmp}|$",
            "$E_{R,ref}$",
            "$|E_{R,ref}-E_{R,cmp}|$",
        ]
        axes[ch, 0].legend(legend_str, loc="upper right", ncol=2, fontsize="x-small")

        # get plot range from peak position
        peak_n = np.argmax(np.max(np.abs(ref_ir), axis=-0))
        peak_range = np.array([peak_n, peak_n + compare_frame_samples])
        peak_range = np.round(peak_range - min([compare_frame_samples / 2, peak_n]))

        # plot TD comparison around peak
        axes[ch, 1].plot(x, ref_ir[ch], linewidth=1, color="black")
        axes[ch, 1].plot(
            x, cmp_ir[ch], linewidth=1.5, alpha=0.75, linestyle="--", color=color
        )
        if ch == diff_ir.shape[0] - 1:
            axes[ch, 1].set_xlabel("Samples")
        axes[ch, 1].set_ylabel("Amplitude")
        axes[ch, 1].tick_params(
            which="both", direction="in", top=True, bottom=True, left=True, right=True
        )
        axes[ch, 1].tick_params(axis="y", labelright=True, labelleft=False)
        axes[ch, 1].set_xticks(
            np.arange(
                peak_n - (compare_frame_samples / 2),
                peak_n + (compare_frame_samples / 2) + 1,
                compare_frame_samples / 4,
            ),
            minor=False,
        )
        axes[ch, 1].set_xticks(np.arange(peak_range[0], peak_range[1], 1), minor=True)
        axes[ch, 1].set_yticks(
            np.arange(_LIM_PEAK_Y[0], _LIM_PEAK_Y[1] + 0.5, 0.5), minor=False
        )
        axes[ch, 1].grid(True, which="major", axis="y", alpha=0.1)
        axes[ch, 1].yaxis.set_label_position("right")
        axes[ch, 1].set_xlim(*peak_range)
        axes[ch, 1].set_ylim(*_LIM_PEAK_Y)
        # axes[ch, 1].set_zorder(-1)
        legend_str = [r"$E_{%s,%s}$" % ("R" if ch else "L", s) for s in ["ref", "cmp"]]
        axes[ch, 1].legend(legend_str, loc="lower right", ncol=2, fontsize="x-small")

        if is_show_fd:
            # plot FD comparison
            # noinspection PyUnboundLocalVariable
            axes[ch, 2].semilogx(freqs, ref_tf[ch], linewidth=1.5, color="black")
            # noinspection PyUnboundLocalVariable
            axes[ch, 2].semilogx(
                freqs,
                cmp_tf[ch],
                linewidth=0.5,
                alpha=0.75,
                linestyle="--",
                color=color,
            )
            # axes[ch, 2].semilogx(freqs, diff_tf[ch], linewidth=0.5, color=color)

            axes[ch, 2].set_xticks(_FREQS_LABELED)
            if ch == diff_ir.shape[0] - 1:
                axes[ch, 2].xaxis.set_major_formatter(
                    FuncFormatter(lambda xx, _: f"{xx / 1000:.16g}")
                )
            axes[ch, 2].set_xlim([20, fs / 2])  # needs to be done after setting xticks
            axes[ch, 2].yaxis.tick_right()
            axes[ch, 2].yaxis.set_label_position("right")
            if ch == diff_ir.shape[0] - 1:
                axes[ch, 2].set_xlabel("Frequency / kHz")
            axes[ch, 2].tick_params(
                which="major",
                direction="in",
                top=True,
                bottom=True,
                left=True,
                right=True,
            )
            axes[ch, 2].tick_params(which="minor", length=0)
            axes[ch, 2].grid(True, which="major", axis="both", alpha=0.25)
            axes[ch, 2].grid(True, which="minor", axis="both", alpha=0.1)
            axes[ch, 2].set_ylabel("Magnitude / dB")
            axes[ch, 2].set_yticks(
                np.arange(*axes[ch, 2].get_ylim(), _STEP_FD_Y), minor=True
            )
            axes[ch, 2].set_ylim(_LIM_FD_Y)
            # axes[ch, 2].set_zorder(-1)

    # remove layout margins
    fig.tight_layout(pad=0)
    # adjust between subplot distances
    plt.subplots_adjust(wspace=0, hspace=0)

    # adjust axes tick labels
    for ch in range(diff_ir.shape[0]):
        _remove_inner_tick_labels(axes, ch, 0)
        _remove_inner_tick_labels(axes, ch, 1)

    return fig


def _generate_table_entry(
    cmp_file_name,
    cmp_azimuth_deg,
    cmp_corr_n_samples,
    cmp_scale_dbfs,
    diff_rms_dbfs,
    diff_max_dbfs,
):
    """
    Generate `pandas.DataFrame` from the provided data. IMPORTANT the variable names in this
    function call are captured and directly applied as virtual column names for the data table.
    HENCE CHANGING A VARIABLE NAME, YOU'LL HAVE TO ALTER IT IN THE ENTIRE BENCHMARKING APPLICATION.

    Parameters
    ----------
    cmp_file_name : str
        reference IR file name
    cmp_azimuth_deg : float
        head rotation azimuth in degrees
    cmp_corr_n_samples : int
        time alignment of reference relative to comparing file in samples
    cmp_scale_dbfs : str
        level alignment of comparing relative to reference file as string of list in dBFS
    diff_rms_dbfs : str
        resulting RMS error of comparison as string of list in dBFS
    diff_max_dbfs : str
        resulting maximum error of comparison as string of list in dBFS

    Returns
    -------
    pandas.DataFrame
        generated data entry, to be appended to the data set (table)
    """
    # collect function parameter names as column names
    params = inspect.signature(_generate_table_entry).parameters.keys()

    # collect function parameter values as data entry
    data = []
    for p in params:  # was not able to write that as a nice one-liner
        data.append(locals()[p])

    # create `pandas.DataFrame`
    return pd.DataFrame([data], columns=params)
