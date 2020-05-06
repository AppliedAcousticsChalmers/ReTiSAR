import sys
from time import sleep

from . import *


_INITIALIZE_DELAY = .5
"""Delay in seconds waited after certain points of the initialization progress to get a clear logging behaviour."""


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
        Create and start a new `HeadTracker` instance, providing head tracking data to a suitable `JackRenderer`
        process.

        Returns
        -------
        HeadTracker
            freshly created instance
        """
        new_tracker = HeadTracker.create_instance_by_type(name + '-Tracker', config.TRACKER_TYPE, config.TRACKER_PORT)
        new_tracker.start()
        sleep(_INITIALIZE_DELAY)

        return new_tracker

    def setup_pre_renderer():
        """
        Create and start a new `JackRenderer` instance, providing a pre-rendering of Array Room Impulse Responses by
        applying a streamed audio signal to it. If applicable, also a matching `JackGenerator` instance will be created
        and started.

        Returns
        -------
        JackRenderer
            freshly created instance
        JackGenerator
            freshly created instance
        """
        try:
            new_renderer = JackRenderer(name + '-PreRenderer', config.BLOCK_LENGTH, config.ARIR_FILE, config.ARIR_TYPE,
                                        sh_max_order=config.SH_MAX_ORDER)
        except ValueError as e:
            logger.error(e)
            logger.error('application interrupted.')
            terminate_all()
            sys.exit(1)

        # check `_counter_dropout` if file was loaded, see `JackRenderer._init_convolver()`
        if new_renderer.get_dropout_counter() is None or tools.transform_into_type(config.HRIR_TYPE, FilterSet.Type)\
                not in [FilterSet.Type.HRIR_MIRO, FilterSet.Type.HRIR_SOFA]:
            logger.warning('skipping microphone array pre-rendering.')
            return None, None

        # in case microphone array audio stream (real-time capture or recorded) should be rendered
        elif tools.transform_into_type(config.ARIR_TYPE, FilterSet.Type) is FilterSet.Type.AS_MIRO:
            logger.warning('skipping microphone array pre-rendering (file still loaded to gather configuration).')

        # in case microphone array IR set should be rendered
        else:
            new_renderer.start(client_connect_target_ports=False)
            new_renderer.set_output_volume_db(config.ARIR_LEVEL)
            new_renderer.set_output_mute(config.ARIR_MUTE)
            sleep(_INITIALIZE_DELAY)

        try:
            new_generator = JackGenerator(name + '-Generator', config.BLOCK_LENGTH,
                                          len(new_renderer.get_client_outputs()),
                                          config.G_TYPE)
        except ValueError:
            logger.warning('skipping generator.')
            return new_renderer, None

        new_generator.start(client_connect_target_ports=False)
        new_generator.set_output_volume_db(config.G_LEVEL)
        new_generator.set_output_mute(config.G_MUTE)
        sleep(_INITIALIZE_DELAY)

        return new_renderer, new_generator

    def setup_renderer(existing_tracker, existing_pre_renderer=None, existing_generator=None):
        """
        Create and start a new `JackRenderer` instance, providing the main signal processing for the application.

        Parameters
        ----------
        existing_tracker : HeadTracker
            beforehand created instance, which provides the process with positional head tracking data
        existing_pre_renderer : JackRenderer, optional
            beforehand created instance, which provides the pre-processed audio input signals
        existing_generator : Jack

        Returns
        -------
        JackRenderer
            freshly created instance
        """
        try:
            new_renderer = JackRenderer(name + '-Renderer', config.BLOCK_LENGTH, config.HRIR_FILE, config.HRIR_TYPE,
                                        config.SOURCE_POSITIONS, existing_tracker.get_shared_position(),
                                        sh_max_order=config.SH_MAX_ORDER, is_main_client=True)
            if config.SH_MAX_ORDER and existing_pre_renderer:
                new_renderer.prepare_renderer_sh_processing(existing_pre_renderer.get_pre_renderer_sh_config(),
                                                            config.ARIR_RADIAL_AMP)

            new_renderer.prepare_input_subsonic(config.HRIR_SUBSONIC)

        except ValueError as e:
            logger.error(e)
            logger.error('application interrupted.')
            terminate_all()
            sys.exit(1)

        new_renderer.start(client_connect_target_ports=False)
        new_renderer.set_output_volume_db(config.HRIR_LEVEL)
        new_renderer.set_output_mute(config.HRIR_MUTE)
        sleep(_INITIALIZE_DELAY)

        if existing_pre_renderer and existing_pre_renderer.is_alive():
            if tools.transform_into_type(config.ARIR_TYPE, FilterSet.Type) is FilterSet.Type.AS_MIRO:
                # connect to system recording ports in case audio stream should be rendered
                if config.SOURCE_FILE:
                    # recorded audio stream (generated `JackPlayer` has to connect to input ports later)
                    new_renderer.client_register_and_connect_inputs(False)
                else:
                    # real-time captured audio stream (connect to system recording ports)
                    new_renderer.client_register_and_connect_inputs(True)
            else:
                new_renderer.client_register_and_connect_inputs(existing_pre_renderer.get_client_outputs())
        if existing_generator:
            new_renderer.client_register_and_connect_inputs(existing_generator.get_client_outputs())

        return new_renderer

    def setup_player(existing_renderer, existing_player=None):
        """
        Create and start a new `JackPlayer` instance as providing audio file playback to the main rendering instance.
        This function can be used to restart the audio file playback (at the end or within the file) by terminating and
        recreating an instance.

        Parameters
        ----------
        existing_renderer : JackRenderer
            beforehand created `JackRenderer` instance, which inputs will be connected from the outputs of this client
        existing_player : JackPlayer, optional
            beforehand existing `JackPlayer` instance, which will be terminated before creating a new one

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
        try:
            new_player = JackPlayer(name + '-Player', config.SOURCE_FILE)

            # check `_counter_dropout` if file was loaded, see `JackPlayer._init_player()`
            if new_player.get_dropout_counter() is not None:
                new_player.start()
                new_player.set_output_volume_db(config.SOURCE_LEVEL)
                new_player.set_output_mute(config.SOURCE_MUTE)
                sleep(_INITIALIZE_DELAY)

                existing_renderer.client_register_and_connect_inputs(new_player.get_client_outputs())

                # start playback if player got restarted
                # (otherwise playback is supposed to be started after all application parts are ready)
                if new_player.is_auto_play and existing_player:
                    new_player.play()

            elif existing_renderer.is_alive():
                if tools.transform_into_type(config.ARIR_TYPE, FilterSet.Type) is FilterSet.Type.AS_MIRO:
                    # connect recording ports to renderer ports directly instead (in case pre-renderer is started)
                    existing_renderer.client_register_and_connect_inputs(True)
                else:
                    # only register renderer ports
                    logger.warning('array IR based rendering was specified without providing a playback source file.')
                    existing_renderer.client_register_and_connect_inputs(None)

            return new_player

        except ValueError as e:
            logger.error(e)
            logger.error('application interrupted.')
            terminate_all()
            sys.exit(1)

    def setup_hp_eq(existing_renderer):
        """
        Create and start a new `JackRenderer` instance, providing a headphone compensation for binaural reproduction.

        Parameters
        ----------
        existing_renderer : JackRenderer
            beforehand created instance, whose outputs will be connected as inputs to this client

        Returns
        -------
        JackRenderer
            freshly created instance
        """
        new_renderer = JackRenderer(name + '-HpEQ', config.BLOCK_LENGTH, config.HPIR_FILE, config.HPIR_TYPE)
        # check `_counter_dropout` if file was loaded, see `JackRenderer._init_convolver()`
        if new_renderer.get_dropout_counter() is None:
            logger.warning('skipping headphone compensation.')
            # connect renderer ports directly to playback instead
            # noinspection PyProtectedMember
            existing_renderer._client_register_and_connect_outputs(True)
        else:
            new_renderer.start(client_connect_target_ports=True)
            new_renderer.set_output_volume_db(config.HPIR_LEVEL)
            new_renderer.set_output_mute(config.HPIR_MUTE)
            sleep(_INITIALIZE_DELAY)

            new_renderer.client_register_and_connect_inputs(existing_renderer.get_client_outputs())

        return new_renderer

    def terminate_all():
        """Terminate all potentially spawned `SubProcess` and remote control clients."""
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
            hp_eq.terminate()
            hp_eq.join()
        except (NameError, AttributeError):
            pass
        try:
            tracker.terminate()
            tracker.join()
        except (NameError, AttributeError):
            pass
        try:
            renderer.terminate()
            renderer.join()
        except (NameError, AttributeError):
            pass

    # use package name as _client name
    name = __package__

    logger = process_logger.setup()
    tracker = setup_tracker()
    pre_renderer, generator = setup_pre_renderer()
    renderer = setup_renderer(tracker, pre_renderer, generator)
    player = setup_player(pre_renderer if (pre_renderer and pre_renderer.is_alive()) else renderer)
    hp_eq = setup_hp_eq(renderer)

    # set tracker reference position at application start
    tracker.set_zero_position()
    # start player playback at application start
    if player and player.is_auto_play:
        player.play()
    sleep(_INITIALIZE_DELAY)

    # startup completed
    print(tools.SEPARATOR)
    logger.info('use [CTRL]+[C] (once!) to interrupt execution or OSC for remote control ...')
    remote = OscRemote(config.REMOTE_OSC_PORT, logger=logger)
    sleep(_INITIALIZE_DELAY)

    # run remote interface until application is interrupted
    try:
        remote.start([player, pre_renderer, generator, hp_eq, tracker, renderer])
    except KeyboardInterrupt:
        logger.error('interrupted by user.')

    # terminate application
    terminate_all()

    return logger
