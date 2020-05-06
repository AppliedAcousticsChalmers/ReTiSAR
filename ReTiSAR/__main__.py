from . import config, tools, __version__

# start execution
tools.parse_cmd_args()
print(tools.SEPARATOR)
if config.STUDY_MODE:
    print('initializing enforced config for study mode ...')
    tools.set_arg(ref=config, arg='LOGGING_PATH', value=None)
    tools.set_arg(ref=config, arg='LOGGING_LEVEL', value='WARNING')
    tools.set_arg(ref=config, arg='IS_PYFFTW_MODE', value=True)
    tools.set_arg(ref=config, arg='PYFFTW_EFFORT', value='FFTW_PATIENT')
    config.LOGGING_PATH = tools.get_absolute_from_relative_package_path(config.LOGGING_PATH)
else:
    # print name, author and license information
    print(open(tools.get_absolute_from_relative_package_path('AUTHORS'), mode='r', encoding='utf-8').read())
    print(f'current version "{__version__}"')
print(tools.SEPARATOR)
tools.request_process_parameters()
print(tools.SEPARATOR)
tools.request_numpy_parameters()
print(tools.SEPARATOR)
if config.IS_PYFFTW_MODE:
    tools.import_fftw_wisdom(is_enforce_load=config.STUDY_MODE)
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
    if config.STUDY_MODE:
        tools.set_arg(ref=_run, arg='_INITIALIZE_DELAY', value=0)
    logger = _run.main()

# end application
print(tools.SEPARATOR)
if config.IS_PYFFTW_MODE and not config.STUDY_MODE:
    tools.export_fftw_wisdom(logger=logger)
    print(tools.SEPARATOR)
logger.info('application ended.') if logger else print('application ended.')
exit(0)
