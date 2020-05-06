from . import config, tools

# start execution
print(tools.SEPARATOR)
tools.parse_cmd_args()
print(tools.SEPARATOR)
tools.request_process_parameters()
print(tools.SEPARATOR)
tools.request_numpy_parameters()
print(tools.SEPARATOR)
if config.IS_PYFFTW_MODE:
    tools.import_fftw_wisdom()
    print(tools.SEPARATOR)

# run in specific mode, if requested by command line parameter
if config.BENCHMARK_MODE:
    from . import _run_benchmark
    logger = _run_benchmark.main()
elif config.VALIDATION_MODE:
    from . import _run_validation
    logger = _run_validation.main()
elif config.DEVELOPER_MODE:
    from . import _run_dev
    logger = _run_dev.main()
else:
    from . import _run
    logger = _run.main()

# end application
print(tools.SEPARATOR)
if config.IS_PYFFTW_MODE:
    tools.export_fftw_wisdom(logger)
    print(tools.SEPARATOR)
logger.info('application ended.') if logger else print('application ended.')
exit(0)
