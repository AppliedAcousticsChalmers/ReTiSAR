import re
from enum import auto, Enum, IntEnum
from time import sleep

import serial

from . import mp_context, tools
from ._subprocess import SubProcess


class HeadTracker(SubProcess):
    """
    Base functionality to run data acquisition from a head tracker contained in a separate
    process. To run the process the functions `start()`, `join()` and `terminate()` have to be
    used.

    Attributes
    ----------
    _position_raw : multiprocessing.Array
        data array containing the absolute head position received directly from the tracker in the
        form of `HeadTracker.DataIndex`, process-safe to be accessed by different functions of
        the tracker
    _position_zero : multiprocessing.Array
        data array containing the reference head position saved position reset in the form of
        `HeadTracker.DataIndex`, process-safe to be accessed by different functions of the tracker
    _position_shared : multiprocessing.Array
        data array containing the relative head position after reset in the form of
        `HeadTracker.DataIndex`, process-safe to be accessed by different subprocesses of this
        application as well
    """

    class Type(Enum):
        """
        Enumeration data type used to get an identification of hardware providing head tracking
        data to the application. It's attributes (with an arbitrary distinct integer value) are
        used as system wide unique constant identifiers.
        """

        POLHEMUS_PATRIOT = auto()
        """
        https://polhemus.com/motion-tracking/all-trackers/patriot

        Used by`HeadTracker` to receive head tracking data over a serial port like
        "/dev/tty.UC-232AC" when adapted to USB. """

        POLHEMUS_FASTRACK = auto()
        """
        https://polhemus.com/motion-tracking/all-trackers/fastrak

        Used by`HeadTracker` to receive head tracking data over a serial port like
        "/dev/tty.UC-232AC" when adapted to USB. """

        RAZOR_AHRS = auto()
        """
        https://github.com/Razor-AHRS/razor-9dof-ahrs/wiki/Tutorial

        Used by`HeadTracker` to receive head tracking data over a serial port like
        "/dev/tty.usbserial-AH03F9XC" when adapted to USB. """

        VRPN = auto()
        """Generic VRPN interface to receive tracking data from a Virtual Reality Peripheral
        Network server application.
        https://github.com/vrpn/vrpn/wiki

        Currently not used or implemented.
        """

        AUTO_ROTATE = auto()
        """Generic functionality to not read tracking data from a device, but generate an
        automatically increasing azimuth angle resulting in a virtually rotating sound scene. """

    class DataIndex(IntEnum):
        """
        Enumeration data type used to get array indices providing storing positional tracking
        data in a single array. It's attributes (with an incrementing distinct integer value) are
        used as system wide unique constant identifiers specifying the index of the according
        data in the array.
        """

        X, Y, Z, AZIM, ELEV, TILT = range(6)

    @staticmethod
    def create_instance_by_type(name, tracker_type, tracker_port, *args, **kwargs):
        """
        Static method to instantiate one of the from `HeadTracker` deriving classes, depending on
        the given `HeadTracker.Type`.

        Parameters
        ----------
        name : str
            name of the spawned tracker process
        tracker_type : str or int
            direct identifier value or string containing the name of one of the provided
            `HeadTracker.Type` members
        tracker_port : str
            system specific path to tracker interface being provided by an appropriate hardware
            driver

        Returns
        -------
        HeadTracker
            created instance according to `HeadTracker.Type`

        Raises
        ------
        ValueError
            in case an unknown tracker type is given
        """
        _type = tools.transform_into_type(tracker_type, HeadTracker.Type)
        if _type is None:
            return HeadTracker(name, tracker_port, *args, **kwargs)
        elif _type == HeadTracker.Type.AUTO_ROTATE:
            return HeadTrackerRotate(name, tracker_port, *args, **kwargs)
        elif _type == HeadTracker.Type.POLHEMUS_PATRIOT:
            return HeadTrackerPatriot(name, tracker_port, *args, **kwargs)
        elif _type == HeadTracker.Type.POLHEMUS_FASTRACK:
            return HeadTrackerFastrack(name, tracker_port, *args, **kwargs)
        elif _type == HeadTracker.Type.RAZOR_AHRS:
            return HeadTrackerRazor(name, tracker_port, *args, **kwargs)

    # noinspection PyTypeChecker
    @staticmethod
    def print_data(array):
        """
        Static method to neatly print positional information in the provided data array with
        style: "[key=value, ...]".

        Parameters
        ----------
        array : multiprocessing.Array
             positional tracking data in the form of `HeadTracker.DataIndex`
        """
        s = "["
        for i in range(len(array)):
            s = f"{s}{HeadTracker.DataIndex(i).name}={array[i]:.1f}"
            if i < len(array) - 1:
                s = f"{s}, "
        return f"{s}]"

    # noinspection PyTypeChecker
    def __init__(self, name, tracker_port, *args, **kwargs):
        """
        Create new instance of a positional head tracker in a separate process.

        Parameters
        ----------
        name : str
            name of the spawned tracker process
        tracker_port : str
            system specific path to tracker interface being provided by an appropriate hardware
            driver
        """
        super().__init__(name, *args, **kwargs)

        self._position_raw = mp_context.Array(
            typecode_or_type="f", size_or_initializer=len(HeadTracker.DataIndex)
        )
        self._position_zero = mp_context.Array(
            typecode_or_type="f", size_or_initializer=len(HeadTracker.DataIndex)
        )
        self._position_shared = mp_context.Array(
            typecode_or_type="f", size_or_initializer=len(HeadTracker.DataIndex)
        )

        self._init_tracker(tracker_port)

        # setting up OSC sender
        self._init_osc_client()

    def _init_tracker(self, _):
        """
        Base function to initialize necessary instances for the tracker process. Deriving classes
        need to override this function to create and initialize the required attributes.
        """
        self._init_config()

    def _init_config(self):
        """
        Base function to set configuration parameters for the tracker process. Deriving classes
        need to override this function to implement the individual parameters.
        """
        self._timeout = 1 / 60  # data send rate

        self._logger.warning("... no tracker will be used.")

    def run(self):
        """
        Overrides the `multiprocessing.Process` function which is automatically called by `start()`
        to run the process. This implementation updates the shared position data with current
        values regularly and makes the process stay alive until the according
        `multiprocessing.Event` is set.
        """
        self._logger.debug("running TRACKER ...")
        try:
            while not self._event_terminate.is_set():
                # individual implementation to read data
                self._read_data()

                # update shared value by current value in a process-safe manner
                with self._position_shared.get_lock():
                    for i in HeadTracker.DataIndex:
                        self._position_shared[i] = (
                            self._position_raw[i] - self._position_zero[i]
                        )

                # communicate current value
                angles_deg = tools.transform_into_wrapped_angles(
                    azim=self._position_shared[HeadTracker.DataIndex.AZIM],
                    elev=self._position_shared[HeadTracker.DataIndex.ELEV],
                    tilt=self._position_shared[HeadTracker.DataIndex.TILT],
                    is_deg=True,
                )
                if self._osc_client:
                    self._osc_client.send_message(
                        f"{self._osc_name}/AzimElevTilt", angles_deg
                    )
                # else:
                #     self._logger.debug(f'azimuth, elevation, tilt degrees {angles_deg}')
        except KeyboardInterrupt:
            self._logger.error("interrupted by user.")

    def _read_data(self):
        """
        Base function to read tracking data from different kinds of hardware. Deriving classes
        need to override this function to implement the actual data acquisition.
        """
        sleep(self._timeout)  # artificial timeout

    def set_zero_position(self):
        """
        Reset the virtual audio scene to be in front where current head rotation is facing.
        Consequently the tracker will always output zeros as the reference position in that exact
        orientation (and cartesian room coordinate).
        """
        for i in HeadTracker.DataIndex:
            self._position_zero[i] = self._position_raw[i]

        self._logger.info(
            f"setting head tracker zero position to {HeadTracker.print_data(self._position_zero)}."
        )

    def set_azimuth_position(self, azim_deg=0.0):
        """
        Parameters
        ----------
        azim_deg : float, optional
            azimuth position in degrees, used to manually set the head rotation in relation to the
            set reference position
        """
        self._position_raw[HeadTracker.DataIndex.AZIM] = (
            self._position_zero[HeadTracker.DataIndex.AZIM] + azim_deg
        )

        self._logger.info(
            f"setting head tracker azimuth position to [AZIM={azim_deg:.1f}]."
        )

    def get_shared_position(self):
        """
        Returns
        -------
        multiprocessing.Array
            Data array always being updated to contain the relative tracked position in the order of
            `HeadTracker.DataIndex`. Other subprocesses of this application can access the data via
            this shared address.
        """
        return self._position_shared


class HeadTrackerRotate(HeadTracker):
    """
    Extended `HeadTracker` implementation for generating pseudo tracking data, to have a moving
    sound scene even without tracker hardware available. This is achieved by incrementing
    positional data, which is utilized best with a rotation in the horizontal plane.

    Attributes
    ----------
    _timeout : float
        virtual refresh rate of tracker, triggering a recalculation of the data again after the
        given wait time
    _position_step : list of float
        step size of data (see `HeadTracker.DataIndex`) being added to the current position every
        `_timeout`
    """

    def _init_config(self):
        """Extends the function of `HeadTracker` to configure individual parameters for the
        pseudo-tracker process."""
        self._timeout = 1 / 60  # data send rate
        self._position_step = [0, 0, 0, 0.5, 0, 0]  # x, y, z, azim, elev, tilt

        self._logger.info(
            f'opened tracker "AUTO_ROTATE"\n'
            f" --> send_rate: {1 / self._timeout} Hz, step_size: {self._position_step[3:]} deg"
        )

    def _read_data(self):
        """
        Extends the function of `HeadTracker` to read data from a tracker hardware, by generating
        pseudo tracking data from incrementing values at a given pace.
        """
        sleep(self._timeout)

        with self._position_raw.get_lock():
            for i in HeadTracker.DataIndex:
                self._position_raw[i] += self._position_step[i]


class HeadTrackerSerial(HeadTracker):
    # noinspection PyUnresolvedReferences
    """
    Extended `HeadTracker` implementation for acquiring data provided via a serial port. Usually
    these are nowadays still easily accessible via USB-to-Serial-adapters, providing a virtual
    serial port when utilizing the correct driver.

    Attributes
    ----------
    _DATA_FIND_FORMAT : str
        regular expression to find desired values in the individual raw data string
    _BAUD_RATE : int
        serial communication port data rate specified for the individual hardware
    _DATA_RAW_INDEX : list of int
        positions of values (see `HeadTracker.DataIndex`) contained in the individual provided raw
        data string
    _DATA_RAW_SCALE : list of float
        scaling factor of values (see `HeadTracker.DataIndex`) contained in the individual provided
        raw data string
    _DATA_INIT_STRING : str, optional
        string sequence that has to be sent to the individual hardware after initialization to
        signal a continuous delivery of tracking data over the serial port
    _serial : serial.Serial
        bidirectional serial data connection to communicate with individual tracking hardware
    _timeout : float, optional
        timeout to keep tracker process alive in case the provided serial port is not available,
        but still providing potential functionality like manipulating the position data by the user
    """

    # noinspection PyTypeChecker
    def _init_tracker(self, port):
        """
        Extends the function of `HeadTracker` to initialize necessary instances for the tracker
        process, by initiating a bidirectional communication with a provided serial port.

        Parameters
        ----------
        port : str
            system specific path to tracker interface being provided by an appropriate hardware
            driver
        """
        # initialize standard parameter configuration
        # matching RegEx could also be "\s*\d+(\s*[-+]?\d*\.?\d*){6}"
        self._DATA_FIND_FORMAT = r"[-+]?\d+\.?\d*"
        self._BAUD_RATE = 0
        self._DATA_RAW_INDEX = [-1] * len(HeadTracker.DataIndex)
        self._DATA_RAW_SCALE = [1.0] * len(HeadTracker.DataIndex)
        self._DATA_INIT_STRING = ""

        # call configuration parameter initialization
        self._init_config()

        try:
            # open serial port
            self._serial = serial.Serial(port=port, baudrate=self._BAUD_RATE, timeout=1)
            self._logger.info(f'opened tracker "{port}"\n --> {self._serial}')

            # signal to start continuous data output mode
            if self._DATA_INIT_STRING and self._DATA_INIT_STRING != "":
                self._serial.write(self._DATA_INIT_STRING.encode())  # as byte sequence
                self._serial.flush()

            # read first data line and discard it since it's often incomplete
            self._serial.readline()
        except serial.SerialException as e:
            self._logger.warning(f"{e.strerror} ... no tracker will be used.")
            self._serial = None
            self._timeout = 1 / 60  # artificial timeout

    def _init_config(self):
        """
        Extends the function of `HeadTracker` to configure individual communication parameters
        according to the used tracker hardware. Deriving classes need to override this function
        to set appropriate parameters.
        """
        raise NotImplementedError(
            f'chosen tracker type "{type(self)}" not implemented yet.'
        )

    def _read_data(self):
        """
        Extends the function of `HeadTracker` to read data from the tracker hardware,
        by line-wise acquisition of byte strings and parsing them in the according manner.
        """
        if not self._serial:
            sleep(self._timeout)
            return

        line = self._serial.readline()
        try:
            line = line.decode().strip()  # get as string
            # self._logger.debug(f'line "{line}"')
            result = re.findall(self._DATA_FIND_FORMAT, line)
            # self._logger.debug(f'result "{result}"')

            for i in HeadTracker.DataIndex:
                if 0 <= self._DATA_RAW_INDEX[i] < len(result):
                    # self._logger.debug(f'get raw {i} from result {self._DATA_RAW_INDEX[i]}')
                    self._position_raw[i] = (
                        float(result[self._DATA_RAW_INDEX[i]]) * self._DATA_RAW_SCALE[i]
                    )
        except UnicodeDecodeError:
            self._logger.warning(f'skipped incomplete TRACKER message "{line}"')

    def terminate(self):
        """
        Extends the function of `HeadTracker` to terminate the tracker process by also closing
        the serial data connection to the hardware.
        """
        if self._serial:
            self._serial.close()

        super().terminate()


class HeadTrackerPatriot(HeadTrackerSerial):
    """
    Individual implementation for Polhemus Patriot hardware, see
    `HeadTracker.Type.POLHEMUS_PATRIOT`.

    example data string          "01     8.938  -41.308    1.130  -95.834    8.138  116.702<CR><LF>"
    contained data               tracker id, x, y, z, azimuth, elevation, tilt, <CR><LF>
    ASCII data byte lengths      60  =  4 + 6x9 + 2

    initialize continuous data mode with byte string "C\r"
    """

    def _init_config(self):
        """Set individual communication parameters according to manual, see `HeadTrackerSerial`."""
        self._BAUD_RATE = 115200
        self._DATA_RAW_INDEX = [1, 2, 3, 4, 5, 6]  # all values after tracker id given
        self._DATA_RAW_SCALE[HeadTracker.DataIndex.AZIM] = -1  # azim inverted
        self._DATA_INIT_STRING = "C\r"


class HeadTrackerFastrack(HeadTrackerSerial):
    """
    Individual implementation for Polhemus Fastrack hardware, see
    `HeadTracker.Type.POLHEMUS_FASTRACK`.

    example data string                 "01   20.40 -61.06  30.01-150.70 -42.08 156.93<CR><LF>"
    contained data                      tracker id, x, y, z, azimuth, elevation, tilt, <CR><LF>
    ASCII data byte lengths             47  =  3 + 6x7 + 2

    initialize continuous data mode with byte string "C"
    """

    def _init_config(self):
        """Set individual communication parameters according to manual, see `HeadTrackerSerial`."""
        self._BAUD_RATE = 115200
        self._DATA_RAW_INDEX = [1, 2, 3, 4, 5, 6]  # all values after tracker id given
        self._DATA_RAW_SCALE[HeadTracker.DataIndex.AZIM] = -1  # azim inverted
        self._DATA_INIT_STRING = "C"


class HeadTrackerRazor(HeadTrackerSerial):
    """
    Individual implementation for Razor AHRS hardware, see `HeadTracker.Type.RAZOR_AHRS`.

    example data string                 "#YPR=24.32,-2.41,154.99"
    contained data                      yaw, pitch, roll ... or ... azimuth, elevation, tilt
    """

    def _init_config(self):
        """Set individual communication parameters according to manual, see `HeadTrackerSerial`."""
        self._BAUD_RATE = 57600
        self._DATA_RAW_INDEX = [-1, -1, -1, 0, 2, 1]  # only azim, tilt, elev given
        self._DATA_RAW_SCALE[HeadTracker.DataIndex.AZIM] = -1  # azim inverted
        self._DATA_RAW_SCALE[HeadTracker.DataIndex.ELEV] = -1  # elev inverted
