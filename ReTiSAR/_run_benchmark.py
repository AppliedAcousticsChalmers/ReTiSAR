import inspect
import logging
import os
import resource
import time

import jack
import matplotlib.pyplot as plt
import pandas as pd

from . import config, process_logger, tools
from ._jack_client import JackClient
from ._jack_renderer import JackRendererBenchmark


def main():
    """
    Function containing the entire benchmarking procedure including defining configuration,
    running the benchmark as well as exporting and visualizing the results.

    Returns
    -------
    logging.Logger
        logger instance

    Raises
    ------
    ValueError
        in case unknown benchmark mode is given
    """
    # ---------------------------------------
    # # BEGIN: DEFINE BENCHMARK CONFIGURATION
    # ---------------------------------------
    _IS_START_OWN_JACK_SERVER = False
    """If own `JackClient`, acting as a JACK background server to prevent it being shutdown and
    started with every first benchmarking client, should be started by this application. """

    _FILTER_LENGTHS = [128, 512, 4096, 110250]
    """Type of FIR filter files to run benchmarks on, keep identical for comparison against old
    runs. """

    _BLOCK_LENGTHS = [128, 256, 512, 1024, 2048, 4096]
    """List of block lengths of the JACK audio server and clients and samples to run benchmarks
    on. """

    _REPETITIONS = 5
    """Number of repetitions per filter file and block length combination benchmarking run. """

    _IS_INCLUDE_OVERHEAD = False
    """If `JackRenderer` level analysis and OSC reporting overheads should be performed during
    benchmark. """
    # ---------------------------------------
    # END:     DEFINE BENCHMARK CONFIGURATION
    # ---------------------------------------

    # there is a limitation of JACK ports per server (256 by default), which leads to a
    # limitation of around 64 JACK clients, when each client has 2 input and 2 output
    # ports
    # TODO: try to avoid port limitation e.g. by starting JACK server with modified command
    #  jackd --port-max 512 -d coreaudio
    _JACK_INSTANCES_LIMIT = 61 if config.BENCHMARK_MODE == "PARALLEL_CLIENTS" else 0

    # prevent error like `OSError: [Errno 24] Too many open files` on OSX
    if (
        config.BENCHMARK_MODE == "PARALLEL_CLIENTS"
        and resource.getrlimit(resource.RLIMIT_NOFILE)[0]
        < config.PROCESS_FILE_LIMIT_MIN
    ):
        import sys

        print(
            f"maximum number of open file descriptors for the current process is smaller then "
            f"{config.PROCESS_FILE_LIMIT_MIN}.\n --> execute `ulimit -n "
            f"{config.PROCESS_FILE_LIMIT_MIN}` in a terminal to manually raise the value!\n"
            f"application interrupted.",
            file=sys.stderr,
        )
        sys.exit(1)

    # use package and file name as _client name
    name = f'{__package__}{os.path.basename(__file__).strip(".py")}'
    if config.BENCHMARK_MODE == "PARALLEL_CONVOLVERS":
        name = f"{name}_convolvers"
    elif config.BENCHMARK_MODE == "PARALLEL_CLIENTS":
        name = f"{name}_clients"
    else:
        raise ValueError(f"unknown type of {config.BENCHMARK_MODE}.")

    # prepare logger
    logger = process_logger.setup(name)

    # create client instance to start JACK server, if it is not running
    if _IS_START_OWN_JACK_SERVER:
        logger.info("starting background client ...")
        background_client = JackClient(
            name=f"{name}-bc",
            is_main_client=True,
            is_disable_file_logger=True,
            is_disable_logger=True,
        )
        background_client.start()
    else:
        logger.warning(
            "skipping creation of a background client. Please make sure you have the JACK server "
            "running already!"
        )

    # run benchmark and capture run time
    run_time = time.time()
    # noinspection PyUnboundLocalVariable
    results, exit_state = _run(
        logger=logger,
        name=name,
        mode=config.BENCHMARK_MODE,
        filter_lengths=_FILTER_LENGTHS,
        block_lengths=_BLOCK_LENGTHS,
        repetitions=_REPETITIONS,
        start_time=run_time,
        is_include_overhead=_IS_INCLUDE_OVERHEAD,
        jack_instances_limit=_JACK_INSTANCES_LIMIT,
    )

    # log runtime as formatted string
    run_time = time.strftime("%Hh%Mm%Ss", time.gmtime(time.time() - run_time))
    time.sleep(0.5)
    if exit_state:
        logger.error(f"... benchmark interrupted after {run_time}.")
    else:
        logger.info(f"... benchmark completed in {run_time}.")

    # terminate background client
    if _IS_START_OWN_JACK_SERVER:
        logger.info("terminating background client ...")
        # noinspection PyUnboundLocalVariable
        background_client.terminate()
        background_client.join(2)
        background_client.close()

    if exit_state:
        # end application
        return logger

    # collect overall benchmark configuration and runtime information
    title = f"{_REPETITIONS} repetitions, {2} channels, {run_time} runtime"
    if _JACK_INSTANCES_LIMIT:
        title = f"{_JACK_INSTANCES_LIMIT} clients limit, {title}"

    # generate and save HTML file
    html_file = process_logger.setup_logfile(name, "html")
    logger.info(f'writing results to "{os.path.relpath(html_file)}" ...')
    tools.export_html(
        html_file,
        results.style.apply(_table_highlight_limit, axis=1).render(),
        title=title,
    )

    # generate and save plots
    results_limits = _get_table_by_query_equal(results, "instances_is_limit", 1)
    plot = _generate_plot(results_limits, title)
    png_file = process_logger.setup_logfile(name, "png")
    logger.info(f'writing results to "{os.path.relpath(png_file)}" ...')
    plot.savefig(png_file, dpi=300)
    pdf_file = process_logger.setup_logfile(name, "pdf")
    logger.info(f'writing results to "{os.path.relpath(pdf_file)}" ...')
    plot.savefig(pdf_file)

    # end application
    # noinspection PyUnresolvedReferences
    logger.info(f'also see results in "{os.path.relpath(logger.file)}".')
    logger.info("... benchmark ended.")
    return logger


def _run(
    logger,
    name,
    mode,
    filter_lengths,
    block_lengths,
    repetitions,
    start_time,
    is_include_overhead,
    jack_instances_limit=0,
):
    """
    Run benchmark loop for the provided configuration. All combinations of parameters are tested,
    hence the time to run grows rapidly, especially when instantiating a high number of clients
    at bigger block lengths.

    Parameters
    ----------
    logger : logging.logger
        beforehand generated logger instance to output meaningful logging information
    name : str
        name inherited to generated JACK client instances
    mode : Mode
        desired benchmarking method
    filter_lengths : List[int]
        list of filter lengths of emulated finite impulse responses used for convolution in the
        generated JACK clients
    block_lengths : List[int]
        list of block lengths used for the convolutions engine (influencing the needed blocks per
        partitioned overlap-save-convolution)
    repetitions : int
        number of repetitions (being averaged afterwards) per combination of `filter_files` and
        `block_lengths`
    is_include_overhead : bool
        if `JackRenderer` level analysis and OSC reporting overheads should be performed
    jack_instances_limit : int, optional
        maximum number of JACK client instances that can be instantiated (this limitation comes
        from the used version of JACK i.e., could be possibly adjusted when rebuilt)

    Returns
    -------
    pandas.DataFrame
        generated data table
    bool
        exit state, True in case process got interrupted
    """
    # prepare empty data table
    results = pd.DataFrame()

    # prepare benchmarking loop
    instances = []
    finished = False
    interrupted = False
    block_lengths_backup = block_lengths.copy()
    filter_lengths_backup = filter_lengths.copy()
    fs = None

    while not finished and not interrupted:
        try:
            # test all desired filter lengths
            while len(filter_lengths):
                filter_len = filter_lengths[0]

                # reset block lengths under test, if benchmark for filter length was completed
                if not len(block_lengths):
                    block_lengths = block_lengths_backup.copy()

                # test all desired blocksizes
                while len(block_lengths):
                    block_len = block_lengths[0]
                    sleep_time = None

                    # test multiple times for average
                    for r in range(repetitions):

                        # continuously start instances
                        instances = []
                        instances_n = 0
                        instances_is_limit = False
                        while not instances_is_limit:
                            # continue execution
                            config.IS_RUNNING.set()

                            # test for JACK client limit
                            if (
                                mode == "PARALLEL_CLIENTS"
                                and jack_instances_limit
                                and instances_n >= jack_instances_limit
                            ):
                                logger.warning(
                                    f"reached JACK client limit at {jack_instances_limit}, "
                                    f"benchmark skipped. "
                                )

                                instances_is_limit = True

                            # instantiate client (only one in case of `Mode.PARALLEL_CONVOLVERS`
                            elif mode == "PARALLEL_CLIENTS" or not instances_n:
                                # create client instance
                                instances.append(
                                    _create_client(
                                        name=f"{name}-c{len(instances):03d}",
                                        filter_len=filter_len,
                                        block_len=block_len,
                                        is_include_overhead=is_include_overhead,
                                    )
                                )
                                # noinspection PyProtectedMember
                                fs = instances[0]._client.samplerate

                                if mode == "PARALLEL_CONVOLVERS":
                                    # Dynamically calculate time, which this process sleeps to
                                    # gather potential audio dropouts of the started clients.
                                    # (Two times) either the filter length or the block size seems
                                    # to be reasonable. This executed only after creating the
                                    # client for once, in case of 'PARALLEL_CONVOLVERS'.
                                    sleep_time = (
                                        filter_len / fs
                                        if filter_len > block_len
                                        else block_len / fs
                                    )
                                    logger.info(
                                        f"using sleep time of {sleep_time * 1000:.0f} ms."
                                    )
                                else:
                                    # It turned out for parallel JACK clients the occurring
                                    # dropouts do not change systematically, if no delay is used.
                                    sleep_time = 0

                            # add convolver instance
                            elif mode == "PARALLEL_CONVOLVERS" and instances_n:
                                instances[0].add_convolver()

                            # increment instance count (clients or convolvers) and wait
                            instances_n += 1
                            time.sleep(sleep_time)

                            # collect and reset dropout counts of the individual clients
                            dropouts_n = _get_and_reset_dropouts(instances)

                            # log current dropout counts and average system load
                            load_mean = round(instances[0].get_cpu_load())
                            dropouts_mean = round(sum(dropouts_n) / len(dropouts_n), 1)

                            if dropouts_mean > 1 / instances_n:
                                instances_is_limit = True
                                lvl = logging.WARNING
                            else:
                                lvl = logging.INFO

                            # save results
                            if sum(dropouts_n) == 0:
                                dropouts_n = [0]
                            results = results.append(
                                _generate_table_entry(
                                    filter_fs=fs,
                                    filter_len=filter_len,
                                    block_len=block_len,
                                    instances_n=instances_n,
                                    instances_is_limit=instances_is_limit,
                                    dropouts_n=dropouts_n,
                                    dropouts_mean=dropouts_mean,
                                    load_mean=load_mean,
                                ),
                                ignore_index=True,
                            )

                            logger.log(
                                lvl,
                                f"dropouts at n={instances_n} with mean={dropouts_mean} "
                                f"{dropouts_n} at load={load_mean:.0f}.",
                            )

                        # log instance number limit
                        logger.warning(
                            f'[{time.strftime("%Hh%Mm%Ss", time.gmtime(time.time() - start_time))}]'
                            f"  filter {len(filter_lengths_backup) - len(filter_lengths) + 1}"
                            f"/{len(filter_lengths_backup)}, "
                            f"block length {len(block_lengths_backup) - len(block_lengths) + 1}"
                            f"/{len(block_lengths_backup)}, "
                            f"repetition {r + 1}/{repetitions}\n"
                            f" --> was not able to run n={instances_n} with block={block_len} of "
                            f"filter={filter_len} at fs={fs}."
                        )
                        time.sleep(0.5)

                        # terminate all started instances
                        logger.info("terminating created instances ...")
                        _terminate_all_clients(instances)
                        print(tools.SEPARATOR)

                    # remove finished block length
                    block_lengths.remove(block_len)

                # remove finished filter files
                filter_lengths.remove(filter_len)

                # end benchmarking
                if not len(filter_lengths):
                    finished = True

        except jack.JackError:
            # noinspection PyUnboundLocalVariable
            logger.error(
                f"JACK error with block={block_len} of filter={filter_len} at fs={fs}."
            )
            logger.info(
                "benchmark for these settings will be repeated and then continued."
            )
        except ValueError as e:
            # end benchmarking
            interrupted = True
            logger.error(
                f"{e}\n --> set JACK server sampling frequency identical to audio file!"
                if "resampling" in e.args[0]
                else e
            )
        except KeyboardInterrupt:
            # end benchmarking
            interrupted = True
            logger.error("interrupted by user.")
        finally:
            logger.info("terminating created instances ...")
            _terminate_all_clients(instances)
            print(tools.SEPARATOR)

    return results, interrupted


def _create_client(name, filter_len, block_len, is_include_overhead):
    """
    Instantiate a JACK client with the provided configuration.

    Parameters
    ----------
    name : str
        name inherited to generated instances
    filter_len : int
        length of filter being emulated for the convolver
    block_len : int
        current block length in samples (determining the number of blocks for the partitioned
        overlap-save-convolution)
    is_include_overhead : bool
        if `JackRenderer` level analysis and OSC reporting overheads should be performed

    Returns
    -------
    JackRendererBenchmark
        generated instance
    """
    r = JackRendererBenchmark(
        name=name,
        block_length=block_len,
        filter_length=filter_len,
        is_single_precision=config.IS_SINGLE_PRECISION,
        ir_trunc_db=None,
        is_detect_clipping=is_include_overhead,
        is_measure_levels=is_include_overhead,
        is_measure_load=is_include_overhead,
        is_main_client=is_include_overhead,
        is_disable_file_logger=True,
        is_disable_logger=True,
    )

    # connect to system recording and playback ports
    r.start(client_connect_target_ports=True)
    r.client_register_and_connect_inputs()

    # ensure system state, do not mute since renderer might not convolve in that case
    r.set_output_mute(False)
    r.set_output_volume_db(-200)
    r.set_client_passthrough(False)

    return r


def _get_and_reset_dropouts(instances):
    """
    Parameters
    ----------
    instances : list of JackClient
        all currently instantiated JACK clients

    Returns
    -------
    list of int
        number of detected audio processing dropouts per client (being set to 0 after reading)
    """
    dropouts_n = []
    for i in instances:
        counter = i.get_dropout_counter()
        with counter.get_lock():
            dropouts_n.append(counter.value)
            counter.value = 0

    return dropouts_n


def _generate_table_entry(
    filter_fs,
    filter_len,
    block_len,
    instances_n,
    instances_is_limit,
    dropouts_n,
    dropouts_mean,
    load_mean,
):
    """
    Generate `pandas.DataFrame` from the provided data. IMPORTANT the variable names in this
    function call are captured and directly applied as virtual column names for the data table.
    HENCE CHANGING A VARIABLE NAME, YOU'LL HAVE TO ALTER IT IN THE ENTIRE BENCHMARKING APPLICATION.

    Parameters
    ----------
    filter_fs : int
        sampling frequency of filter in Hz
    filter_len : int
        length of filter in samples
    block_len : int
        length of block in samples
    instances_n : int
        number of parallel instances
    instances_is_limit : bool
        if limit of parallel instances was exceeded
    dropouts_n : list of int
        number of detected audio dropouts per instance
    dropouts_mean : float
        average number of detected audio dropouts of all instances
    load_mean : int
        average system load of all instances

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


def _terminate_all_clients(instances):
    """
    Parameters
    ----------
    instances : list of JackClient
        all currently instantiated JACK clients
    """
    # pause execution
    config.IS_RUNNING.clear()
    time.sleep(0.5)  # needed to not crash when terminating clients
    # noinspection PyTypeChecker
    for i in instances:
        i.terminate()
        i.join(2)  # needed to not crash when terminating clients
        i.close()
    # noinspection PyUnresolvedReferences
    instances.clear()
    time.sleep(1)  # safety buffer


def _table_highlight_limit(entry):
    """
    Format a data entry in the provided data collection. In this case an entire data row is colored
    red, in case the `instances_is_limit` attribute is true.

    Parameters
    ----------
    entry : pandas.Series
        existing data set (table)

    Returns
    -------
    str
        CSS formatting depending on the given criteria
    """
    is_limit = entry["instances_is_limit"]
    return ["color: red" if is_limit else "" for _ in entry]


def _get_table_by_query_equal(table, column_name, value):
    """
    Return a new data set, whose elements out of the provided data set fulfill the condition: value
    of attribute `column_name` is equal to `value`.

    Parameters
    ----------
    table : pandas.DataFrame
        existing data set (table)
    column_name : str
        name of virtual table column to be tested
    value : int or str
        value to be tested

    Returns
    -------
    pandas.DataFrame
        data set with applied query condition
    """
    return table.copy().query(f"{column_name}=={value}")


def _generate_plot(results_limits, title=None):
    """
    Generate a results plot visualizing the determined number of possible JACK clients in parallel
    for all different combinations of configuration settings.

    Parameters
    ----------
    results_limits : pandas.DataFrame
        data set (table) containing only those entries with the determined client number limits
    title : str, optional
        headline being printed above the plot

    Returns
    -------
    matplotlib.Figure
        generated plot figure
    """
    # # BEGIN: DEFINE PLOTTING CONFIGURATION
    _BLOCK_LENGTHS_PLOT = [128, 256, 512, 1024, 2048, 4096]
    # replace block lengths by 0,1,2,... for linear x axis while plotting
    results_limits.block_len.replace(
        _BLOCK_LENGTHS_PLOT, range(len(_BLOCK_LENGTHS_PLOT)), inplace=True
    )
    results_grouped = results_limits.groupby(["filter_fs", "filter_len"])

    _GROUP_POINT_SPACING_X = len(results_grouped) * 0.025  # in plot units
    _GROUP_TEXT_SPACING_Y = -10  # in points
    # get an incrementing group index for plot offset
    # groups: 2 -> [-0.5,+0.5]  3 -> [-1,0,+1]  ...
    _GROUP_MARKER_OFFSET = -0.5 * len(results_grouped) + 0.5
    # # END: DEFINE PLOTTING CONFIGURATION

    # get limits for y axis (space at the bottom for limits as text)
    _PLOT_Y_MAX = results_limits.instances_n.max()
    _PLOT_Y_MIN = len(results_grouped) * _PLOT_Y_MAX / -20
    _PLOT_Y_GRID = int(_PLOT_Y_MAX / 50)
    if _PLOT_Y_GRID < 1:
        _PLOT_Y_GRID = 1
    _PLOT_Y_TICK = _PLOT_Y_GRID * 3 if _PLOT_Y_MAX >= 15 else 1

    # generate figure and plot individual data
    group_index = -1
    figure = plt.figure()
    for _, group in results_grouped:
        group_index += 1
        label = f"{group.filter_len.iloc[0]} samples @ {group.filter_fs.iloc[0] / 1000:.1f} kHz"

        # collect median values for each 'block_len'
        x = []
        y = []
        for _, g in group.groupby("block_len"):
            x.append(g.block_len.iloc[0])
            y.append(g.instances_n.median() - 1)

        # plot median values connected by line
        line = plt.plot(x, y, label=label)
        color = line[0].get_color()

        # plot median values as text
        for x, y in zip(x, y):
            plt.annotate(
                f"{y:.0f}",
                (x, _PLOT_Y_MIN),
                xytext=(
                    0,
                    _GROUP_TEXT_SPACING_Y * (group_index - len(results_grouped)),
                ),
                textcoords="offset points",
                ha="center",
                va="top",
                annotation_clip=False,
                color=color,
            )

        # plot all raw values for each 'block_len' with scaling markers
        for _, g in group.groupby("block_len"):
            unique_counts = g.instances_n.value_counts()

            # x position per block_len including marker offset
            x = [
                g.block_len.iloc[0]
                + (group_index + _GROUP_MARKER_OFFSET) * _GROUP_POINT_SPACING_X
            ] * len(unique_counts)

            # y position per unique count
            y = unique_counts.index - 1

            # marker size per unique count with scaling
            s = [10 * v for v in unique_counts.values]

            plt.scatter(x, y, s=s, marker="2", color=color)

    # add title
    if title:
        plt.title(title, fontsize="large")

    # set overall plot settings
    plt.legend(loc="best", fontsize="x-small")
    plt.grid(which="both", linestyle="-", alpha=0.4)
    plt.minorticks_on()

    # set x axis related plot settings
    plt.xlabel("block_len")
    plt.gca().xaxis.grid(False, which="minor")
    plt.xlim(-0.5, len(_BLOCK_LENGTHS_PLOT) - 0.5)
    plt.gca().set_xticks(range(len(_BLOCK_LENGTHS_PLOT)))
    plt.gca().set_xticks([], minor=True)
    plt.gca().set_xticklabels(_BLOCK_LENGTHS_PLOT)

    # set y axis related plot settings
    plt.ylabel("instances_n - 1   (raw and median)")
    plt.ylim(_PLOT_Y_MIN, _PLOT_Y_MAX)
    plt.gca().set_yticks(list(range(0, _PLOT_Y_MAX + 1, _PLOT_Y_TICK)))
    plt.gca().set_yticks(list(range(0, _PLOT_Y_MAX + 1, _PLOT_Y_GRID)), minor=True)

    return figure
