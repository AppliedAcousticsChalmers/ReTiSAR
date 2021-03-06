from enum import auto, IntEnum

import numpy as np
from scipy.signal import sos2tf, sosfilt, sosfilt_zi, tf2sos

from . import config, tools
from ._jack_client import JackClient


class JackGenerator(JackClient):
    """
    Extended functionality from `JackClient` to also provide the real-time generation artificial
    audio signal into the JACK output ports. To run the process the functions `start()`,
    `join()` and `terminate()` have to be used.

    Attributes
    ----------
    _generator : Generator
        instance of utilized sound generator
    """

    def __init__(
        self, name, block_length, output_count, generator_type, *args, **kwargs
    ):
        """
        Extends the `JackClient` function to initialize a new JACK client and process. According
        to the documentation all attributes must be initialized in this function, to be available
        to the spawned process.

        Parameters
        ----------
        name : str
            name of the JACK client and spawned process
        block_length : int
            system specific size of every audio block
        output_count : int
            number of output channels
        generator_type : str or Generator.Type
            type of audio signal generator
        """
        super().__init__(name=name, block_length=block_length, *args, **kwargs)

        dtype = np.float32 if self._is_single_precision else np.float64
        self._generator = Generator.create_instance_by_type(
            generator_type=generator_type, output_count=output_count, dtype=dtype
        )

        if config.IS_DEBUG_MODE:
            self._debug_generate_block()

        # plot
        gen = self._generator.generate_block(block_length=self._client.blocksize)
        name = f"{self._logger.name}_{generator_type}_{block_length}_{output_count}ch"
        tools.export_plot(
            figure=tools.plot_ir_and_tf(
                gen[:8], fs=self._client.samplerate, set_fd_db_y=50, step_db_y=10
            ),
            name=name,
            logger=self._logger,
        )

    def start(self, client_connect_target_ports=True):
        """
        Extends the `JackClient` function to `start()` the process. Here also the function
        concerning the JACK output ports suitable for binaural rendering is called.

        Parameters
        ----------
        client_connect_target_ports : jack.Ports or bool, optional
            see `_client_register_and_connect_outputs()` for documentation
        """
        super().start()

        self._logger.debug("activating JACK client ...")
        self._client.activate()
        self._client_register_and_connect_outputs(client_connect_target_ports)
        self._event_ready.set()

    def _client_register_and_connect_outputs(self, target_ports=True):
        """
        Register a number of output ports according to the used `Generator` instance to the
        current client in case none existed before. For further behaviour see documentation see
        called overridden function of `JackClient`.
        """
        if not self._generator:
            return

        # register output ports if none exist
        if not len(self._client.outports):
            # noinspection PyProtectedMember
            self._client_register_outputs(self._generator._output_count)

        # noinspection PyProtectedMember
        super()._client_register_and_connect_outputs(target_ports)

    def _debug_generate_block(self):
        """
        Provides debugging possibilities the `generate_block()` function before running as a
        separate process, where breakpoints do not work anymore.

        Returns
        -------
        numpy.ndarray
            generated output blocks in time domain of size [number of output channels;
            `_block_length`]
        """
        op1 = self._generator.generate_block(block_length=self._client.blocksize)
        tools.plot_ir_and_tf(op1[:8], fs=self._client.samplerate)
        # potentially check output block here!

        op2 = self._generator.generate_block(block_length=self._client.blocksize)
        tools.plot_ir_and_tf(op2[:8], fs=self._client.samplerate)
        # in case of IIR colored noise specifically, check that first generated blocks are
        # unimpaired!

        return op2

    def _process(self, _):
        """
        Process block of audio data. This implementation provides the generation of an artificial
        audio signal by a `Generator`.

        Returns
        -------
        numpy.ndarray
            generated block of audio data that will be delivered to JACK
        """
        if not config.IS_RUNNING.is_set():
            return None
        return self._generator.generate_block(block_length=self._client.blocksize)


class Generator(object):
    """
    Flexible structure used to generate artificial audio signals for an arbitrary number of
    channels. A given `Generator.Type` defines what and how individual signals will be generated
    and provided to a `JackGenerator` instance.

    Attributes
    ----------
    _output_count : int
        number of channels to generate
    """

    class Type(IntEnum):
        """
        Enumeration data type used to get an identification of generators utilizing a certain
        algorithm to generate artificial audio signals i.e., noise with a specified coloration.
        It's attributes (with an distinct integer value) are used as system wide unique constant
        identifiers.

        The given numbers are relevant in case of the auto-regressive noise generation algorithm,
        meaning each color corresponds to an inverse frequency power in the noise power density
        spectrum.
        """

        NOISE_AR_PURPLE = -2
        NOISE_AR_BLUE = -1
        NOISE_AR_PINK = 1
        NOISE_AR_BROWN = 2
        NOISE_WHITE = auto()
        NOISE_IIR_PINK = auto()
        NOISE_IIR_EM = auto()
        NOISE_IIR_EIGENMIKE = NOISE_IIR_EM
        IMPULSE_DIRAC = auto()

    @staticmethod
    def create_instance_by_type(generator_type, output_count, dtype):
        """
        Parameters
        ----------
        generator_type : str or Generator.Type
            type of audio signal generator
        output_count : int
            number of channels to generate
        dtype : str or numpy.dtype or type
            data type to generate

        Returns
        -------
        Generator
            created instance according to `JackGenerator.Type`

        Raises
        ------
        ValueError
            in case unknown generator type is given
        """
        _type = tools.transform_into_type(generator_type, Generator.Type)
        if _type is Generator.Type.IMPULSE_DIRAC:
            return GeneratorImpulse(output_count=output_count, dtype=dtype)
        elif _type is Generator.Type.NOISE_WHITE:
            return GeneratorNoise(output_count=output_count, dtype=dtype)
        elif _type is Generator.Type.NOISE_IIR_PINK:
            return GeneratorNoiseIir(
                output_count=output_count, dtype=dtype, color="PINK"
            )
        elif _type in [Generator.Type.NOISE_IIR_EM, Generator.Type.NOISE_IIR_EIGENMIKE]:
            return GeneratorNoiseIir(output_count=output_count, dtype=dtype, color="EM")
        elif _type in [
            Generator.Type.NOISE_AR_PURPLE,
            Generator.Type.NOISE_AR_BLUE,
            Generator.Type.NOISE_AR_PINK,
            Generator.Type.NOISE_AR_BROWN,
        ]:
            return GeneratorNoiseAr(output_count=output_count, dtype=dtype, power=_type)
        else:
            raise ValueError(
                f'unknown generator type "{_type}", see `JackGenerator.Type` for reference!'
            )

    def __init__(self, output_count, dtype):
        """
        Parameters
        ----------
        output_count : int
            number of channels to generate
        dtype : str or numpy.dtype or type
            data type to generate
        """
        self._output_count = output_count
        self._dtype = dtype

    def generate_block(self, block_length, is_transposed=None):
        """
        Individual implementation of an algorithm to generate audio signals in desired shape.

        Parameters
        ----------
        block_length : int
            number of samples to generate
        is_transposed : bool, optional
            if generated data should have the Fortran style of memory arrangement of size
            [number of samples; number of output channels]

        Returns
        -------
        numpy.ndarray
            generated block of audio data
        """
        raise NotImplementedError(
            "This function needs to be overridden by deriving classes."
        )


class GeneratorImpulse(Generator):
    """
    Extended `Generator` implementation for generating dirac impulses. This means a block of
    zeros will be generated and every first sample set to 1.
    """

    def generate_block(self, block_length, is_transposed=None):
        """
        Individual implementation of an algorithm to generate audio signals in desired shape.

        Parameters
        ----------
        block_length : int
            number of samples to generate
        is_transposed : bool, optional
            if generated data should have the Fortran style of memory arrangement of size
            [number of samples; number of output channels]

        Returns
        -------
        numpy.ndarray
            generated block of audio data
        """
        shape = (self._output_count, block_length)
        if is_transposed:
            shape = shape[::-1]  # invert

        impulse = np.zeros(shape=shape, dtype=self._dtype)
        if is_transposed:
            impulse[0] = 1
        else:
            impulse[:, 0] = 1

        return impulse


class GeneratorNoise(Generator):
    """
    Extended `Generator` implementation for generating white noise. This means an incoherent
    block of noise will be generated for every output channel.
    """

    def generate_block(self, block_length, is_transposed=False):
        """
        Individual implementation of an algorithm to generate audio signals in desired shape.

        Parameters
        ----------
        block_length : int
            number of samples to generate
        is_transposed : bool, optional
            if generated data should have the Fortran style of memory arrangement of size
            [number of samples; number of output channels]

        Returns
        -------
        numpy.ndarray
            generated block of audio data
        """
        shape = (self._output_count, block_length)
        if is_transposed:
            shape = shape[::-1]  # invert

        return tools.generate_noise(shape=shape, dtype=self._dtype)


class GeneratorNoiseAr(GeneratorNoise):
    """
    Extended `GeneratorNoise` implementation for generating noise with a desired coloration. This
    means an incoherent block of noise will be generated for every output channel.

    This implementation uses an auto-regressive algorithm, mimicking the application of IIR
    filters of very high order. The current implementation is computationally very expensive,
    since the samples are acquired in time domain, hence the implementation can not be utilized
    in real-time so far.

    Attributes
    ----------
    _coefficients : numpy.ndarray
        filter coefficients for auto-regressive algorithm according to given order and power
    _buffer : numpy.ndarray
        constantly shifting buffer for auto-regressive algorithm of size [number of coefficients;
        number of output channels]

    References
    ----------
        N. J. Kasdin, “Discrete simulation of colored noise and stochastic processes and
        1/f^α power law noise generation,” Proceedings of the IEEE, vol. 83, no. 5, pp. 802–827,
        May 1995. :doi:`10.1109/5.381848`
    """

    def __init__(self, output_count, dtype, power, order=8):
        """
        Parameters
        ----------
        output_count : int
            number of channels to generate
        dtype : str or numpy.dtype or type
            data type to generate
        power : int
            power indicating the type of coloration for auto-regressive method
        order : int, optional
            order indicating the mimicked filter order for auto-regressive method
        """
        super().__init__(output_count=output_count, dtype=dtype)

        coefficients = np.zeros(shape=order, dtype=dtype)
        coefficients[0] = 1
        for k in range(1, order):
            coefficients[k] = (k - 1 - power / 2) * coefficients[k - 1] / k

        self._coefficients = coefficients[1:].copy()
        self._buffer = np.zeros(
            shape=(self._coefficients.shape[0], self._output_count),
            dtype=self._coefficients.dtype,
        )

    def generate_block(self, block_length, is_transposed=False):
        """
        Individual implementation of an algorithm to generate audio signals in desired shape.

        Parameters
        ----------
        block_length : int
            number of samples to generate
        is_transposed : bool, optional
            if generated data should have the Fortran style of memory arrangement of size
            [number of samples; number of output channels]

        Returns
        -------
        numpy.ndarray
            generated block of audio data
        """
        # generate white noise
        normal = super().generate_block(block_length=block_length, is_transposed=True)

        # run auto-regressive filtering
        for n in normal:
            n -= np.dot(self._coefficients, self._buffer)
            self._buffer = np.roll(self._buffer, shift=1, axis=0)
            self._buffer[0] = n

        if is_transposed:
            return normal
        else:
            return normal.T


class GeneratorNoiseIir(GeneratorNoise):
    """
    Extended `GeneratorNoise` implementation for generating noise with a desired coloration. This
    means an incoherent block of noise will be generated for every output channel.

    This implementation uses filtering with provided IIR filter coefficients. For better numeric
    stability this filter is employed by IIR second-order sections (instead of B-A-coefficients).

    Attributes
    ----------
    _GAIN_FACTOR : float
        linear multiplicand to reach a comparable output range like `GeneratorNoise`
    _B_PINK : numpy.ndarray
        IIR filter numerator coefficients to achieve pink noise coloration from [1]
    _A_PINK : numpy.ndarray
        IIR filter denominator coefficients to achieve pink noise coloration from [1]
    _SOS_EM : numpy.ndarray
        IIR filter second-order section coefficients to achieve Eigenmike noise coloration from [2]
    _sos : numpy.ndarray
        utilized IIR filter second-order section coefficients according to chosen coloration

    References
    ----------
    .. [1] https://ccrma.stanford.edu/~jos/sasp/Example_Synthesis_1_F_Noise.html
    .. [2] H. Helmholz, D. Lou Alon, S. V. A. Garí, and J. Ahrens, “Instrumental Evaluation of
           Sensor Self-Noise in Binaural Rendering of Spherical Microphone Array Signals,” in Forum
           Acusticum, 2020, pp. 1–8.
    """

    def __init__(self, output_count, dtype, color):
        """
        Parameters
        ----------
        output_count : int
            number of channels to generate
        dtype : str or numpy.dtype or type
            data type to generate
        color : str
            coloration of noise to generate
        """
        super().__init__(output_count=output_count, dtype=dtype)
        color = color.upper()

        # initialize constants
        self._GAIN_FACTOR = 20

        # fmt: off
        self._B_PINK = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786], dtype=dtype)
        self._A_PINK = np.array([1, -2.494956002, 2.017265875, -0.522189400], dtype=dtype)

        # self._SOS_EM = np.array(
        #     [[0.642831971019108, -1.285660835867235, 0.642829227003050, 1, -1.938240831077712,
        #       0.938773412606562],
        #      [1.026898172219099, -1.958636131748593, 0.936872530871250, 1, -1.998609306155938,
        #       0.998614711257974],
        #      [1.016158122207582, - 2, 0.984300636295923, 1, -1.869587739664868,
        #       0.874716652825408],
        #      [2, 0.311116397866933, 0.032678524446224, 1, -1.199612825442955, 0.209446907520534],
        #      [0.025112532019265, -0.022708413470341, 0.002848575141982, 1, -0.041478045261128,
        #       0.084631884608049]],
        #     dtype=dtype
        # )  # according to Fig. 5a (+30 dB gain) in [2]
        self._SOS_EM = np.array(
            [[1.005691483788964, -2, 0.994309737062589, 1, -1.995131712857242, 0.995141919519996],
             [0.053750499045410, -0.070551854851138, 0.018848551742250, 1, -1.396203288062881,
              0.421364316283091],
             [1.000002228996281, -2, 0.999998915874074, 1, -1.999189537733665, 0.999193631613767],
             [2, -0.050188084384710, 0.000794285722317, 1, -0.173018215643467, 0.053670187934911],
             [1.383915599293111, 1.640929515677082, 0.383739206962885, 1, 1.074403226778343,
              0.393608533940758]],
            dtype=dtype
        )  # according to Fig. 5b (-10 dB gain) in [2]
        # fmt: on

        # pick utilized coefficients
        if color == "PINK":
            a = self._A_PINK
            self._sos = tf2sos(b=self._B_PINK, a=self._A_PINK).astype(dtype)
            # "It is generally discouraged to convert from TF to SOS format, since doing so
            # usually will not improve numerical precision errors."
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.tf2sos.html
        elif color in ["EM", "EIGENMIKE"]:
            [_, a] = sos2tf(sos=self._SOS_EM)
            self._sos = self._SOS_EM
            # "Consider designing filters in ZPK format and converting directly to SOS."
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.tf2sos.html
        else:
            raise NotImplementedError(
                f'chosen noise generator color "{color}" not implemented yet.'
            )

        # initialize IIR filter delay conditions
        self._last_delays = sosfilt_zi(sos=self._sos).astype(dtype)
        # adjust in middle axis according to output count
        self._last_delays = np.repeat(
            self._last_delays[:, np.newaxis, :], repeats=output_count, axis=1
        )

        # approximate decay time to skip transient response part of IIR filter, according to [1]
        t60_samples = int(np.log(1000.0) / (1.0 - np.abs(np.roots(a)).max())) + 1
        # generate and discard long enough sequence to skip IIR transient response (decay time)
        _ = self.generate_block(block_length=t60_samples)

    def generate_block(self, block_length, is_transposed=False):
        """
        Individual implementation of an algorithm to generate audio signals in desired shape.

        Parameters
        ----------
        block_length : int
            number of samples to generate
        is_transposed : bool, optional
            if generated data should have the Fortran style of memory arrangement of size
            [number of samples; number of output channels]

        Returns
        -------
        numpy.ndarray
            generated block of audio data
        """
        # generate white noise
        normal = super().generate_block(
            block_length=block_length, is_transposed=is_transposed
        )

        # transpose initial IIR filter delay conditions in case output should be transposed
        # this is not nice, but should only happens
        if is_transposed and self._last_delays.shape[1] != normal.shape[1]:
            self._last_delays = self._last_delays.T

        # filter signal along time axis
        [shaped, self._last_delays] = sosfilt(
            sos=self._sos,
            x=normal,
            axis=0 if is_transposed else 1,
            zi=self._last_delays,
        )

        # apply gain
        return shaped * self._GAIN_FACTOR
