import inspect
import sys

from pythonosc import dispatcher, osc_server

from . import tools


# noinspection PyTypeChecker
class OscRemote(object):
    """
    Open Sound Control server to receive OSC messages that will invoke mapped client functions.
    http://opensoundcontrol.org/

    The OSC message to e.g. set activate mute state of a `JackClient` will look like this:
    /renderer/mute 1

    Also look at the provided PureData example showing more examples to control different kind of
    clients.

    Attributes
    ----------
    _port : int
        port of OSC server to receive messages
    _server : osc_server.ThreadingOSCUDPServer
        instance of OSC server
    _logger : logging.Logger or None
        instance to provide identical logging behaviour as the calling process
    """

    def __init__(self, port, logger=None):
        """
        Parameters
        ----------
        port : int
            port of OSC server to receive messages
        logger : logging.Logger, optional
            instance to provide identical logging behaviour as the calling process
        """
        self._port = port
        self._logger = logger

        self._server = None

    def start(self, clients):
        """
        Start OSC server to receive messages for all mapped functions regarding the provided
        clients. In the current implementation running the OSC server blocks further application
        execution, , keeping the application alive until the server is released.

        Parameters
        ----------
        clients : list of SubProcess
            clients that receive a function mapping (see implementation for individual functions)
            so they will be remote controllable
        """

        def add_mapping(command, function_name, is_reversible=True):
            """
            Add mapping from an OSC target to a function. The function will be invoked with all
            additionally provided parameters as soon as the matching OSC command is received.

            Parameters
            ----------
            command : str
                OSC target command which will be used in combination with the client name to invoke
                the mapped function
            function_name : str
                name of the clients function that will be invoked later
            is_reversible : bool, optional
                if OSC target will be mapped twice, so the order of the command and the client name
                does not matter
            """
            # check if client has the named function
            if not (
                hasattr(client, function_name)
                and callable(getattr(client, function_name))
            ):
                return

            # add mapping
            dispatch.map(
                f"{client_name}/{command}",
                OscRemote.handle_function,
                client,
                function_name,
                self._logger,
            )
            if is_reversible:
                dispatch.map(
                    f"{command}/{client_name}",
                    OscRemote.handle_function,
                    client,
                    function_name,
                    self._logger,
                )

        # generate mapping for all clients
        dispatch = dispatcher.Dispatcher()
        for client in clients:
            if not client:
                continue

            # prepare simplified client name as OSC target
            client_name = tools.transform_into_osc_target(client.name)

            # see `JackClient` if not otherwise specified
            add_mapping("mute", "set_output_mute")
            add_mapping("volume", "set_output_volume_db")
            add_mapping("volume_relative", "set_output_volume_relative_db")
            add_mapping("volume_port_relative", "set_output_volume_port_relative_db")
            add_mapping("delay", "set_input_delay_ms")
            add_mapping("crossfade", "set_client_crossfade")  # see `JackRenderer`
            add_mapping("passthrough", "set_client_passthrough")  # see `JackRenderer`
            add_mapping("order", "set_renderer_sh_order")  # see `JackRenderer`
            add_mapping("zero", "set_zero_position")  # see `HeadTracker`
            add_mapping("azimuth", "set_azimuth_position")  # see `HeadTracker`
            add_mapping("stop", "stop")  # see `JackPlayer`
            add_mapping("play", "play")  # see `JackPlayer`

            # TODO: implement `JackPlayer` restarting if playback was finished
            # player = _setup_player(
            #     pre_renderer
            #     if (pre_renderer and pre_renderer.is_alive())
            #     else renderer,
            #     player,
            # )

        # add mapping to terminate
        dispatch.map("/quit", OscRemote.handle_terminate, self)
        dispatch.map("/exit", OscRemote.handle_terminate, self)

        # TODO: This line fails in case multiple ReTiSAR instances are run, due an OSC server
        #  already running (the same port being already bound). In the second instance, either try
        #  to reference the already existing OSC server or create a new server with an available
        #  port number!?
        # start OSC server
        self._server = osc_server.ThreadingOSCUDPServer(
            ("127.0.0.1", self._port), dispatch
        )
        log_str = f"listening to OSC messages at {self._server.server_address} ..."
        self._logger.info(log_str) if self._logger else print(log_str)

        # run OSC server and block further execution, keeping the application alive until the
        # server is released
        self._server.serve_forever()

    def terminate(self):
        """
        Shutdown OSC server if it is running and in the current implementation releasing the
        application to continue execution (most likely to shutdown all child processes and
        terminating itself.
        """
        if self._server is None:
            return

        log_str = "terminating OSC server ..."
        self._logger.info(log_str) if self._logger else print(log_str)
        self._server.shutdown()
        self._server = None

    @staticmethod
    def handle_terminate(_, references, *__):
        """
        Call the terminate function of this instance to shutdown the application.

        Parameters
        ----------
        _ : str
            OSC target
        references : list of object
            instance, in this case a reference to itself
        __ : any
            ignored parameters
        """
        self: OscRemote = references[0]

        log_str = "terminated by user."
        self._logger.error(log_str) if self._logger else print(log_str, file=sys.stderr)
        self.terminate()

    # noinspection PyUnresolvedReferences
    @staticmethod
    def handle_function(_, references, *parameters):
        """
        Call the intended function of the instance with the parameters provided. Additional
        parameters will be dropped, in case there are more given then the referenced function
        takes.

        Parameters
        ----------
        _ : str
            OSC target
        references : list of SubProcess, str, logging.Logger
            instance and function that should be invoked as well as reference to logger to provide
            identical logging behaviour as calling process
        parameters : tuple of {int, float, str}
            parameters provided after the OSC target
        """
        client = references[0]
        function_name = references[1]
        logger = references[2]

        function_parameters_count = len(
            inspect.signature(getattr(client, function_name)).parameters
        )
        if len(parameters) > function_parameters_count:
            log_str = (
                f'skipping overhang OSC parameters "'
                f'{", ".join(str(p) for p in parameters[function_parameters_count:])}".'
            )
            logger.warning(log_str) if logger else print(log_str, file=sys.stderr)
            parameters = parameters[:function_parameters_count]

        if len(parameters) == 0:
            # log_str = f'calling ... {type(client).__name__}.{function_name}()'
            # logger.info(log_str) if logger else print(log_str)
            getattr(client, function_name)()
        else:
            # log_str = "calling ... {}.{}({})".format(
            #     type(client).__name__, function_name, *parameters
            # )
            # logger.info(log_str) if logger else print(log_str)
            getattr(client, function_name)(*parameters)
