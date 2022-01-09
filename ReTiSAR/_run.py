from time import sleep

from . import (
    config,
    FilterSet,
    HeadTracker,
    JackGenerator,
    JackPlayer,
    JackRenderer,
    OscRemote,
    process_logger,
    tools,
)

_INITIALIZE_DELAY = 0.5
"""Delay in seconds waited after certain points of the initialization progress to get a clear
logging behaviour. """


def main():
    """
    Function containing the real-time rendering procedure realizing different configurations.

    Returns
    -------
    logging.Logger
        logger instance
    """

    def setup_tracker():
        """
        Create and start a new `HeadTracker` instance, providing head tracking data to a suitable
        `JackRenderer` process.

        Returns
        -------
        HeadTracker
            freshly created instance
        """
        new_tracker = HeadTracker.create_instance_by_type(
            name=f"{name}-Tracker",
            tracker_type=config.TRACKER_TYPE,
            tracker_port=config.TRACKER_PORT,
        )
        new_tracker.start()
        sleep(_INITIALIZE_DELAY)

        return new_tracker

    def setup_pre_renderer():
        """
        Create and start a new `JackRenderer` instance, providing a pre-rendering of Array Room
        Impulse Responses by applying a streamed audio signal to it. If applicable, also a matching
        `JackGenerator` instance will be created and started.

        Returns
        -------
        JackRenderer
            freshly created instance
        JackGenerator
            freshly created instance
        """
        new_renderer = None
        new_generator = None
        try:
            new_renderer = JackRenderer(
                name=f"{name}-PreRenderer",
                block_length=config.BLOCK_LENGTH,
                filter_name=config.ARIR_FILE,
                filter_type=config.ARIR_TYPE,
                sh_max_order=config.SH_MAX_ORDER,
                sh_is_enforce_pinv=config.SH_IS_ENFORCE_PINV,
                ir_trunc_db=config.IR_TRUNCATION_LEVEL,
                is_main_client=False,
                is_single_precision=config.IS_SINGLE_PRECISION,
            )

            # check `_counter_dropout` if file was loaded, see `JackRenderer._init_convolver()`
            if new_renderer.get_dropout_counter() is None or tools.transform_into_type(
                config.HRIR_TYPE, FilterSet.Type
            ) not in [FilterSet.Type.HRIR_MIRO, FilterSet.Type.HRIR_SOFA]:
                logger.warning("skipping microphone array pre-rendering.")
                return None, new_generator

            # in case of microphone array audio stream (real-time capture or recorded)
            elif (
                tools.transform_into_type(config.ARIR_TYPE, FilterSet.Type)
                is FilterSet.Type.AS_MIRO
            ):
                logger.warning(
                    "skipping microphone array pre-rendering (file still loaded to gather "
                    "configuration). "
                )

            # in case microphone array IR set should be rendered
            else:
                new_renderer.start(client_connect_target_ports=False)
                new_renderer.set_output_volume_db(config.ARIR_LEVEL)
                new_renderer.set_output_mute(config.ARIR_MUTE)
                sleep(_INITIALIZE_DELAY)
        except (ValueError, FileNotFoundError, RuntimeError) as e:
            logger.error(e)
            terminate_all(additional_renderer=new_renderer)
            raise InterruptedError

        try:
            new_generator = JackGenerator(
                name=f"{name}-Generator",
                block_length=config.BLOCK_LENGTH,
                output_count=len(new_renderer.get_client_outputs()),
                generator_type=config.G_TYPE,
                is_main_client=False,
                is_single_precision=config.IS_SINGLE_PRECISION,
            )

            new_generator.start(client_connect_target_ports=False)
            new_generator.set_output_volume_db(config.G_LEVEL)
            new_generator.set_output_volume_relative_db(config.G_LEVEL_REL)
            new_generator.set_output_mute(config.G_MUTE)
            sleep(_INITIALIZE_DELAY)
        except ValueError:
            logger.warning("skipping generator.")
        return new_renderer, new_generator

    def setup_renderer(
        existing_tracker, existing_pre_renderer=None, existing_generator=None
    ):
        """
        Create and start a new `JackRenderer` instance, providing the main signal processing for
        the application.

        Parameters
        ----------
        existing_tracker : HeadTracker
            beforehand created instance, which provides the process with positional head tracking
            data
        existing_pre_renderer : JackRenderer, optional
            beforehand created instance, which provides the pre-processed audio input signals
        existing_generator : JackGenerator, optional
            beforehand created `JackGenerator` instance, which outputs will be connected to the
            inputs of this client

        Returns
        -------
        JackRenderer
            freshly created instance
        """
        new_renderer = None
        try:
            new_renderer = JackRenderer(
                name=f"{name}-Renderer",
                block_length=config.BLOCK_LENGTH,
                filter_name=config.HRIR_FILE,
                filter_type=config.HRIR_TYPE,
                input_delay_ms=config.HRIR_DELAY,
                source_positions=config.SOURCE_POSITIONS,
                shared_tracker_data=existing_tracker.get_shared_position(),
                sh_max_order=config.SH_MAX_ORDER,
                sh_is_enforce_pinv=config.SH_IS_ENFORCE_PINV,
                ir_trunc_db=config.IR_TRUNCATION_LEVEL,
                is_main_client=True,
                is_measure_levels=True,
                is_single_precision=config.IS_SINGLE_PRECISION,
            )
            if config.SH_MAX_ORDER is not None and existing_pre_renderer:
                new_renderer.prepare_renderer_sh_processing(
                    input_sh_config=existing_pre_renderer.get_pre_renderer_sh_config(),
                    mrf_limit_db=config.ARIR_RADIAL_AMP,
                    compensation_type=config.SH_COMPENSATION_TYPE,
                )

            new_renderer.start(client_connect_target_ports=False)
            new_renderer.set_output_volume_db(config.HRIR_LEVEL)
            new_renderer.set_output_mute(config.HRIR_MUTE)
            sleep(_INITIALIZE_DELAY)

            if existing_pre_renderer and existing_pre_renderer.is_alive():
                if (
                    tools.transform_into_type(config.ARIR_TYPE, FilterSet.Type)
                    is FilterSet.Type.AS_MIRO
                ):
                    # connect to system recording ports in case audio stream should be rendered
                    if config.SOURCE_FILE:
                        # recorded audio stream (generated `JackPlayer` has to connect to input
                        # ports later)
                        new_renderer.client_register_and_connect_inputs(
                            source_ports=False
                        )
                    else:
                        # real-time captured audio stream (connect to system recording ports)
                        new_renderer.client_register_and_connect_inputs(
                            source_ports=True
                        )
                else:
                    new_renderer.client_register_and_connect_inputs(
                        existing_pre_renderer.get_client_outputs()
                    )
            if existing_generator:
                new_renderer.client_register_and_connect_inputs(
                    existing_generator.get_client_outputs()
                )
        except (ValueError, FileNotFoundError, RuntimeError) as e:
            logger.error(e)
            terminate_all(additional_renderer=new_renderer)
            raise InterruptedError
        return new_renderer

    def setup_player(existing_renderer, existing_player=None):
        """
        Create and start a new `JackPlayer` instance as providing audio file playback to the main
        rendering instance. This function can be used to restart the audio file playback (at the
        end or within the file) by terminating and recreating an instance.

        Parameters
        ----------
        existing_renderer : JackRenderer
            beforehand created `JackRenderer` instance, which inputs will be connected from the
            outputs of this client
        existing_player : JackPlayer, optional
            beforehand existing `JackPlayer` instance, which will be terminated before creating a
            new one

        Returns
        -------
        JackPlayer
            freshly created instance
        """
        if existing_player:
            if existing_player.is_alive():
                existing_player.play()
                return existing_player

            existing_player.terminate()
            existing_player.join()

        new_player = None
        try:
            new_player = JackPlayer(
                name=f"{name}-Player",
                file_name=config.SOURCE_FILE,
                is_auto_play=config.SOURCE_IS_AUTO_PLAY,
                block_length=config.BLOCK_LENGTH,
                is_main_client=False,
                is_single_precision=config.IS_SINGLE_PRECISION,
            )

            # check `_counter_dropout` if file was loaded, see `JackPlayer._init_player()`
            if new_player.get_dropout_counter() is not None:
                new_player.start()
                new_player.set_output_volume_db(config.SOURCE_LEVEL)
                new_player.set_output_mute(config.SOURCE_MUTE)
                sleep(_INITIALIZE_DELAY)

                existing_renderer.client_register_and_connect_inputs(
                    new_player.get_client_outputs()
                )

                # start playback if player got restarted (otherwise playback is supposed to be
                # started after all application parts are ready)
                if new_player.is_auto_play and existing_player:
                    new_player.play()

            elif existing_renderer.is_alive():
                if (
                    tools.transform_into_type(config.ARIR_TYPE, FilterSet.Type)
                    is FilterSet.Type.AS_MIRO
                ):
                    # connect recording ports to renderer ports directly instead (in case
                    # pre-renderer is started)
                    existing_renderer.client_register_and_connect_inputs(
                        source_ports=True
                    )
                else:
                    # only register renderer ports
                    logger.warning(
                        "array IR based rendering was specified without providing a playback "
                        "source file. "
                    )
                    existing_renderer.client_register_and_connect_inputs(
                        source_ports=None
                    )
        except (ValueError, FileNotFoundError, RuntimeError) as e:
            logger.error(e)
            terminate_all(additional_renderer=new_player)
            raise InterruptedError
        return new_player

    def setup_hpcf(existing_renderer):
        """
        Create and start a new `JackRenderer` instance, providing a headphone compensation for
        binaural reproduction.

        Parameters
        ----------
        existing_renderer : JackRenderer
            beforehand created instance, whose outputs will be connected as inputs to this client

        Returns
        -------
        JackRenderer
            freshly created instance
        """
        new_renderer = None
        try:
            new_renderer = JackRenderer(
                name=f"{name}-HPCF",
                block_length=config.BLOCK_LENGTH,
                filter_name=config.HPCF_FILE,
                filter_type=config.HPCF_TYPE,
                ir_trunc_db=config.IR_TRUNCATION_LEVEL,
                is_main_client=False,
                is_single_precision=config.IS_SINGLE_PRECISION,
            )

            # check `_counter_dropout` if file was loaded, see `JackRenderer._init_convolver()`
            if new_renderer.get_dropout_counter() is None:
                logger.warning("skipping headphone compensation.")
                # connect renderer ports directly to playback instead
                # noinspection PyProtectedMember
                existing_renderer._client_register_and_connect_outputs(
                    target_ports=True
                )
            else:
                new_renderer.start(client_connect_target_ports=True)
                new_renderer.set_output_volume_db(config.HPCF_LEVEL)
                new_renderer.set_output_mute(config.HPCF_MUTE)
                sleep(_INITIALIZE_DELAY)

                new_renderer.client_register_and_connect_inputs(
                    existing_renderer.get_client_outputs()
                )
        except (ValueError, FileNotFoundError, RuntimeError) as e:
            logger.error(e)
            terminate_all(additional_renderer=new_renderer)
            raise InterruptedError
        return new_renderer

    def terminate_all(additional_renderer=None):
        """
        Terminate all (potentially) spawned child processes after muting the last client in the
        rendering chain (headphone compensation or binaural renderer).

        Parameters
        ----------
        additional_renderer : JackRenderer, optional
            renderer that should be terminated, which was not (yet) returned to the main run
            function and is part of the usually implemented list of clients (see below)
        """
        config.IS_RUNNING.clear()
        try:
            additional_renderer.terminate()
            additional_renderer.join()
        except (NameError, AttributeError):
            pass
        try:
            if hpcf.is_alive():
                hpcf.set_output_mute(True)
            else:
                renderer.set_output_mute(True)
        except (NameError, AttributeError):
            pass
        try:
            remote.terminate()
        except (NameError, AttributeError):
            pass
        try:
            player.terminate()
            player.join()
        except (NameError, AttributeError):
            pass
        try:
            pre_renderer.terminate()
            pre_renderer.join()
        except (NameError, AttributeError):
            pass
        try:
            generator.terminate()
            generator.join()
        except (NameError, AttributeError):
            pass
        try:
            hpcf.terminate()
            hpcf.join()
        except (NameError, AttributeError):
            pass
        try:
            tracker.terminate()
            tracker.join()
        except (NameError, KeyError, AttributeError):
            pass
        try:
            renderer.terminate()
            renderer.join()
        except (NameError, AttributeError):
            pass

    # use package name as _client name
    name = __package__

    # pause execution
    config.IS_RUNNING.clear()

    logger = process_logger.setup()
    try:
        tracker = setup_tracker()
        pre_renderer, generator = setup_pre_renderer()
        renderer = setup_renderer(
            existing_tracker=tracker,
            existing_pre_renderer=pre_renderer,
            existing_generator=generator,
        )
        player = setup_player(
            existing_renderer=pre_renderer
            if (pre_renderer and pre_renderer.is_alive())
            else renderer
        )
        hpcf = setup_hpcf(existing_renderer=renderer)
    except InterruptedError:
        logger.error("application interrupted.")
        terminate_all()
        return logger  # terminate application

    # set tracker reference position at application start
    tracker.set_zero_position()
    # start player playback at application start
    if player and player.is_auto_play:
        player.play()
    sleep(_INITIALIZE_DELAY)

    # start execution of all clients
    config.IS_RUNNING.set()

    # startup completed
    print(tools.SEPARATOR)
    logger.info(
        "use [CTRL]+[C] (once!) to interrupt execution or OSC for remote control ..."
    )
    remote = OscRemote(config.REMOTE_OSC_PORT, logger=logger)
    sleep(_INITIALIZE_DELAY)

    # run remote interface until application is interrupted
    try:
        remote.start(clients=[player, pre_renderer, generator, hpcf, tracker, renderer])
    except KeyboardInterrupt:
        logger.error("interrupted by user.")

    # terminate application
    terminate_all()
    return logger
