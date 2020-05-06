import inspect
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import fmin

from . import *


# def main_snr():
#     """
#     Function containing the entire validation procedure including defining configuration, running the validation
#     against provided reference impulse response set as well as exporting and visualizing the results.
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
#     _NOISE_IN_FILE = 'log/SNR_4096_noise_in.wav'
#     _NOISE_OUT_FILE = 'log/SNR_4096_noise_out.wav'
#     _SIGNAL_IN_FILE = 'log/SNR_4096_signal_in.wav'
#     _SIGNAL_OUT_FILE = 'log/SNR_4096_signal_out.wav'
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
#     name = __package__ + os.path.basename(__file__).strip('.py')
#     logger = process_logger.setup(name)
#
#     # read recorded files
#     noise_in_td, fs = soundfile.read(_NOISE_IN_FILE, dtype=np.float32)
#     noise_out_td, fs = soundfile.read(_NOISE_OUT_FILE, dtype=np.float32)
#     signal_in_td, fs = soundfile.read(_SIGNAL_IN_FILE, dtype=np.float32)
#     signal_out_td, fs = soundfile.read(_SIGNAL_OUT_FILE, dtype=np.float32)
#
#     # truncate recordings to shortest length
#     length = min([noise_in_td.shape[0], noise_out_td.shape[0], signal_in_td.shape[0], signal_out_td.shape[0]])
#     noise_in_td = noise_in_td[:length, 0].T
#     noise_out_td = noise_out_td[:length, 0].T
#     signal_in_td = signal_in_td[:length, 0].T
#     signal_out_td = signal_out_td[:length, 0].T
#
#     # calculate differences
#     noise_diff_td = np.fft.irfft(np.abs(np.fft.rfft(noise_out_td, axis=-1))
#                                  / np.abs(np.fft.rfft(noise_in_td, axis=-1)), axis=-1)
#     signal_diff_td = np.fft.irfft(np.abs(np.fft.rfft(signal_out_td, axis=-1))
#                                   / np.abs(np.fft.rfft(signal_in_td, axis=-1)), axis=-1)
#
#     # generate plots
#     noise_stack_td = np.vstack([noise_in_td, noise_out_td, noise_diff_td])
#     signal_stack_td = np.vstack([signal_in_td, signal_out_td, signal_diff_td])
#     _plot_result(os.path.splitext(os.path.relpath(_NOISE_OUT_FILE))[0], noise_stack_td, fs, logger=logger)
#     _plot_result(os.path.splitext(os.path.relpath(_SIGNAL_OUT_FILE))[0], signal_stack_td, fs, logger=logger)
#
#     return logger


# noinspection PyProtectedMember
def main():
    """
    Function containing the entire validation procedure including defining configuration, running the validation against
    provided reference impulse response set as well as exporting and visualizing the results.

    Returns
    -------
    logging.Logger
        logger instance
    """
    # ----------------------------------------
    # # BEGIN: DEFINE VALIDATION CONFIGURATION
    # ----------------------------------------
    _CMP_FILES_DIR = 'res/validation'
    """Path of directory being scanned for comparative impulse response sets."""
    _TRUNC_VOLUME_DB = -100
    """Volume in dB the shorter impulse response set will be truncated to before comparison after it reached the last
    sample over the specified level."""
    _REF_TRUNC_SAMPLES = 512
    """Number of samples of end of the reference impulse response set that will be ignored when analysing the 
    truncation length by `_TRUNC_VOLUME_DB`.
    
    This is used since the reference implementation of `sound_field_analysis-py` generates impulse responses which have
    a small rise at the very end of the response."""
    _IS_REF_NORM_INDIVIDUALLY = True
    """If reference IR channels should be normalized individually for better comparability of RMS error."""
    # ----------------------------------------
    # END:     DEFINE VALIDATION CONFIGURATION
    # ----------------------------------------

    # prepare logger
    name = __package__ + os.path.basename(__file__).strip('.py')
    logger = process_logger.setup(name)

    # prepare empty data table
    results = pd.DataFrame()

    # check provided comparative IRs azimuth offset
    cmp_azimuth_offset = config.SOURCE_POSITIONS
    if len(cmp_azimuth_offset) > 1:
        logger.error('given comparative IRs azimuth offset "{}" does not contain one position tuple.'.format(
            cmp_azimuth_offset))
        return logger
    elif len(cmp_azimuth_offset) == 1 and cmp_azimuth_offset[0][0]:
        cmp_azimuth_offset = cmp_azimuth_offset[0][0]  # only keep azimuth of first tuple
        logger.warning('[INFO]  considering azimuth offset {} degree with all comparative IRs ...'.format(
            cmp_azimuth_offset))
    else:
        cmp_azimuth_offset = 0

    # check provided reference IR
    ref_file = tools.get_absolute_from_relative_package_path(config.VALIDATION_MODE)
    if not os.path.isfile(ref_file):
        logger.error('given reference IR "{}" not found.'.format(ref_file))
        return logger

    # load reference IR set and normalize
    ref_set = FilterSet.create_instance_by_type(ref_file, FilterSet.Type.HRIR_SSR)
    if _IS_REF_NORM_INDIVIDUALLY:
        ref_set.load(block_length=None, logger=logger, is_prevent_resampling=True, is_normalize_individually=True)
    else:
        ref_set.load(block_length=None, logger=logger, is_prevent_resampling=True, is_normalize=True)
    fs = ref_set._fs

    # load and analyze all comparative IR sets
    _CMP_FILES_DIR = tools.get_absolute_from_relative_package_path(_CMP_FILES_DIR)
    for cmp_file, cmp_azimuth in zip(*_parse_files_and_azimuth(_CMP_FILES_DIR, 'Impulse', '.wav', cmp_azimuth_offset)):
        print(tools.SEPARATOR)

        # load current comparative IR set
        cmp_file = tools.get_absolute_from_relative_package_path(cmp_file)
        cmp_set = FilterSet.create_instance_by_type(cmp_file, FilterSet.Type.HRIR_SSR)
        cmp_set.load(block_length=None, logger=logger, is_prevent_resampling=True, is_normalize=False, check_fs=fs)

        # only keep IRs
        logger.info('selecting IRs from head orientation azimuth {:.1f} deg, elevation {:.1f} deg.'.format(
            cmp_azimuth, 0))
        ref_ir = ref_set.get_filter_td(azim_deg=cmp_azimuth, elev_deg=0)
        cmp_ir = cmp_set.get_filter_td()
        if ref_ir.shape[0] != cmp_ir.shape[0]:
            logger.error('given IRs have different number of channels ({} and {}).'.format(
                ref_ir.shape[0], cmp_ir.shape[0]))
            return logger

        # truncate shorter filter
        if ref_ir.shape[-1] < cmp_ir.shape[-1]:
            if _REF_TRUNC_SAMPLES:
                # ignore last samples of reference IR
                ref_ir = ref_ir[:, :-_REF_TRUNC_SAMPLES]
                logger.info('ignored "{}" last {}.'.format(
                    os.path.relpath(ref_file), tools.get_pretty_delay_str(_REF_TRUNC_SAMPLES, fs)))

            ref_ir = _trunc_ir_out(ref_file, ref_ir, fs, _TRUNC_VOLUME_DB, logger=logger)
        else:
            cmp_ir = _trunc_ir_out(cmp_file, cmp_ir, fs, _TRUNC_VOLUME_DB, logger=logger)

        # get time alignment, check for being identical for every channel
        cmp_corr_n = np.zeros(ref_ir.shape[0], dtype=np.int32)
        for ch in range(len(cmp_corr_n)):
            cmp_corr_n[ch] = np.argmax(np.correlate(cmp_ir[ch], ref_ir[ch], mode='valid'))
        # check match of correlation times of all channels
        if (cmp_corr_n == cmp_corr_n[0]).all():
            cmp_corr_n = cmp_corr_n[0]
        else:
            # arir = sfa.io.read_miro_struct(_ARIR_REF_FILE)
            # _ARIR_RADIUS = float(arir.configuration.scatter_radius)
            _ARIR_RADIUS = 0.08749999850988388  # in meters, from 'res/ARIR/CR1_VSA_110RS_L_struct'
            delay = ((cmp_corr_n[0] - cmp_corr_n[1]) / 2) / fs  # in seconds
            angle = delay * tools.SPEED_OF_SOUND / _ARIR_RADIUS  # before inverse sine
            log_str = 'time alignment is not identical for all channels ({}).\n' \
                      ' --> i.e. as per ear offset resulting from horizontal head rotation ({}).\n' \
                      ' --> i.e. with {:.1f} cm head / array radius'. \
                format(tools.get_pretty_delay_str(cmp_corr_n, fs),
                       tools.get_pretty_delay_str((cmp_corr_n[0] - cmp_corr_n[1]) / 2, fs),
                       _ARIR_RADIUS * 100)
            if np.abs(angle) > 1:
                logger.error(log_str + ' such a large offset could not be explained by an azimuth rotation.')
            else:
                logger.error(
                    log_str + ' might be explained by an azimuth rotation offset of approx. {:.1f} deg.'.format(
                        np.rad2deg(np.arcsin(angle))))

            # # generate plot to visualize different time alignment
            # ir_plot = np.vstack(
            #     [np.correlate(cmp_ir[ch], ref_ir[ch], mode='valid')[cmp_corr_n.min() - 100:cmp_corr_n.max() + 100]
            #      for ch in range(len(cmp_corr_n))])
            # tools.export_plot(tools.plot_ir_and_tf(ir_plot, fs, is_share_y=False), 'corr_mismatch', logger=logger)
            #
            # for ch in range(len(cmp_corr_n)):
            #     ir_plot = np.vstack([ref_ir[ch, :256], cmp_ir[ch, cmp_corr_n[ch]:cmp_corr_n[ch] + 256]])
            #     tools.export_plot(tools.plot_ir_and_tf(ir_plot, fs, is_share_y=False),
            #                       'corr_mismatch_ch{}'.format(ch), logger=logger)

            return logger

        # align and truncate to reference IR
        cmp_ir = cmp_ir[:, cmp_corr_n:cmp_corr_n + ref_ir.shape[-1]].copy()
        logger.info('time aligned "{}" to {}.'.format(
            os.path.relpath(cmp_file), tools.get_pretty_delay_str(cmp_corr_n, fs)))

        # stack channels for equal level alignment
        ref_ir_stack = np.hstack([ch for ch in ref_ir])
        cmp_ir_stack = np.hstack([ch for ch in cmp_ir])
        # perform least RMS fit and level alignment
        cmp_scale = fmin(func=_calculate_rms_error, x0=1, args=(cmp_ir_stack, ref_ir_stack),
                         disp=False, full_output=True)
        cmp_scale_db = 20 * np.log10(cmp_scale[0][0])
        logger.info('level aligned "{}" by {:+.2f} dB.\n'
                    ' --> current function value: {:f}, iterations: {:d}, function evaluations: {:d}.'.
                    format(os.path.relpath(cmp_file), cmp_scale_db, cmp_scale[1], cmp_scale[2], cmp_scale[3]))
        # only keep RMS fit result
        cmp_scale = cmp_scale[0]
        cmp_ir *= cmp_scale

        # calculate difference
        diff_ir = ref_ir - cmp_ir
        diff_rms = 20 * np.log10(tools.calculate_rms(diff_ir))
        diff_ir[np.nonzero(diff_ir == 0)] = np.nan  # prevent zeros
        diff_max = np.nanmax(20 * np.log10(np.abs(diff_ir)), axis=-1)  # ignore NaNs
        logger.info('calculated RMS difference is {} dBFS.'.format(
            np.array2string(diff_rms, precision=2)))

        # plot difference (cut file extension as name)
        name = os.path.splitext(os.path.relpath(cmp_file))[0]
        tools.export_plot(_plot_result(ref_ir, cmp_ir), name, logger=logger, file_type='png+pdf')

        # save results
        results = results.append(_generate_table_entry(os.path.relpath(cmp_file), cmp_azimuth, cmp_corr_n,
                                                       np.array2string(cmp_scale_db, precision=2),
                                                       np.array2string(diff_rms, precision=2),
                                                       np.array2string(diff_max, precision=2)), ignore_index=True)

    print(tools.SEPARATOR)
    # generate and save HTML file
    # noinspection PyUnboundLocalVariable
    html_title = '"{}" reference, {:d} Hz, {:d} samples (truncated at {:.1f} dB)'.\
        format(os.path.relpath(ref_file), fs, ref_ir.shape[-1], _TRUNC_VOLUME_DB)
    html_file = os.path.join(_CMP_FILES_DIR, os.path.splitext(os.path.basename(ref_file))[0] + os.path.extsep + 'html')
    logger.info('writing results to "{}" ...'.format(os.path.relpath(html_file)))
    tools.export_html(html_file, results.style.render(), title=html_title)

    # end application
    logger.info('... validation ended.')
    return logger


def _parse_files_and_azimuth(path, starting, ending, azimuth_offset=0):
    files = []
    azimuths = []
    for r, _, f in os.walk(path):
        for file in f:
            if file.startswith(starting) and file.endswith(ending):
                files.append(os.path.join(r, file))
                azimuths.append(float(file.strip(ending).split('_')[-1]) + azimuth_offset)
    # return files[:1], azimuths[:1]
    return files, azimuths


def _trunc_ir_out(file_name, ir, fs, volume, logger):
    # get level under peak as amplitude
    amp_trunc = np.abs(ir).max() * 10 ** (volume / 20)

    # get first sample over cutout amplitude from the back
    n_trunc_rev = np.argmax(np.flip(np.abs(ir), axis=-1) > amp_trunc) - 1

    # truncate IR if index found
    if n_trunc_rev > 0:
        ir = ir[:, :-n_trunc_rev]
        logger.info('truncated "{}" to {} dB under peak at {}.'.format(
            os.path.relpath(file_name), -volume, tools.get_pretty_delay_str(ir.shape[-1], fs)))
    else:
        logger.info('kept "{}" at {}.'.format(
            os.path.relpath(file_name), tools.get_pretty_delay_str(ir.shape[-1], fs)))

    return ir


def _calculate_rms_error(scale, ir_scaled, ir):
    return tools.calculate_rms(ir - (scale * ir_scaled))


def _plot_result(ref_ir, cmp_ir, compare_frame_samples=128):
    """
    Parameters
    ----------
    ref_ir : numpy.ndarray
        time domain (real) reference data
    cmp_ir : numpy.ndarray
        time domain (real) comparative data
    compare_frame_samples : int
        number of samples that should be plotted IR around peak (reference is 0th order)

    Returns
    -------
    matplotlib.figure.Figure
        generated plot
    """
    # calculate difference
    diff_ir = ref_ir - cmp_ir

    # prevent zeros
    diff_ir[np.nonzero(diff_ir == 0)] = np.nan
    # transform td data into logarithmic scale
    diff_ir = 20 * np.log10(np.abs(diff_ir))

    fig, axes = plt.subplots(nrows=diff_ir.shape[0], ncols=2, squeeze=False,
                             sharex='col', sharey='col')
    for ch in range(diff_ir.shape[0]):
        # select color and plot string
        color = 'C{:d}'.format(ch)
        legend_str = [r'$E_{%s,%s}$' % ('R' if ch else 'L', s) for s in ['ref', 'cmp']]

        # plot difference
        axes[ch, 0].plot(np.arange(0, len(diff_ir[ch])), diff_ir[ch], linewidth=.5, color=color)
        axes[ch, 0].set_xlim(-1, len(diff_ir[ch]) + 1)
        if ch == diff_ir.shape[0] - 1:
            axes[ch, 0].set_xlabel('Samples')
        axes[ch, 0].set_ylabel(r'$20*log| $%s$ \,-\, $%s$ |$ in dBFS' % (legend_str[0], legend_str[1]))
        axes[ch, 0].tick_params(direction='in', top=True, bottom=True, left=True, right=True)

        # get plot range from peak position
        peak_n = np.argmax(np.abs(ref_ir), axis=-1)
        peak_range = peak_n.mean()
        peak_range = np.array([peak_range, peak_range + compare_frame_samples])
        peak_range = np.round(peak_range - min([compare_frame_samples / 2, peak_n.mean()]))
        if (peak_n < peak_range[0]).any() or (peak_n > peak_range[1]).any():
            raise ValueError('IR peaks {} outside of given comparison plot frame {}.'.format(
                np.array2string(peak_n),  np.array2string(peak_range)))

        # plot comparison around peak
        axes[ch, 1].plot(np.arange(0, len(ref_ir[ch])), ref_ir[ch], linewidth=1.5, color='black')
        axes[ch, 1].plot(np.arange(0, len(cmp_ir[ch])), cmp_ir[ch], linewidth=1.5, linestyle='--', color=color)
        axes[ch, 1].set_xlim(*peak_range)
        if ch == diff_ir.shape[0] - 1:
            axes[ch, 1].set_xlabel('Samples')
        axes[ch, 1].set_ylabel('Amplitude')
        axes[ch, 1].tick_params(direction='in', top=True, bottom=True, left=True, right=True)
        axes[ch, 1].tick_params(axis='y', labelright=True, labelleft=False)
        axes[ch, 1].yaxis.set_label_position('right')

        # plot legend
        axes[ch, 1].legend(legend_str, loc='lower right', ncol=2, fontsize='x-small')

    # remove layout margins
    fig.tight_layout(pad=0)

    return fig


def _generate_table_entry(cmp_file_name, cmp_azimuth_deg, cmp_corr_n_samples, cmp_scale_dbfs, diff_rms_dbfs,
                          diff_max_dbfs):
    """
    Generate `pandas.DataFrame` from the provided data. IMPORTANT the variable names in this function call are captured
    and directly applied as virtual column names for the data table. HENCE CHANGING A VARIABLE NAME, YOU'LL HAVE TO
    ALTER IT IN THE ENTIRE BENCHMARKING APPLICATION.

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
