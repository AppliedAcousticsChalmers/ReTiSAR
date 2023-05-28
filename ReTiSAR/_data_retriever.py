import os.path
import sys
from time import sleep


class DataRetriever(object):
    """
    Flexible interface to check the availability of requested project resources. Since resources
    cannot be distributed as part of this publication, only links to publicly available resources
    can be provided. When a certain recourse is requested on startup of a rendering component,
    its availability or otherwise existence of a respective source reference is checked.
    """

    _EXTENSION = "source"
    """File extension used for the reference files."""

    @staticmethod
    def retrieve(path, is_download=True, logger=None):
        """
        Check for existence of the requested resource. Depending on the provided file (data or
        source), different steps will be performed in order to allow the application to find the
        data file, or allow the user to give instruction to provide the data file.

        In the current implementation the data will be attempted to download and unpacked in case
        a zip archive is available. All steps are logged and shown to the user. The startup
        process will be interrupted in case the data file is not available i.e., downloading and
        / or unpacking of the data failed.

        Parameters
        ----------
        path : str or numpy.ndarray
            path to requested resource (data or source file) or directly provided filter
            coefficients where latter are returned directly
        is_download : bool, optional
            if file should be downloaded in case no data but a source file is available
        logger : logging.logger, optional
            instance to provide identical logging behaviour as the calling process

        Returns
        -------
        str or numpy.ndarray
            path to resource data file (not necessarily available) or filter coefficients in case
            they were directly provided
        """
        if not isinstance(path, str):
            return path

        data = DataRetriever._get_data_path(path)
        if DataRetriever.has_data(path):
            if data != path:
                log_str = (
                    f'source file "{os.path.relpath(path)}" given, but data is already '
                    f"available."
                )
                logger.warning(log_str) if logger else print(
                    f"[WARNING]  {log_str}", file=sys.stderr
                )

        elif DataRetriever.has_source(path):
            source = DataRetriever._get_source_path(path)
            if path != source:
                log_str = (
                    f'not yet available data file "{os.path.relpath(path)}" given, but '
                    f"source file was found."
                )
            else:
                log_str = (
                    f'source file "{os.path.relpath(path)}" given, but data is not yet '
                    f"available."
                )
            logger.warning(log_str) if logger else print(
                f"[WARNING]  {log_str}", file=sys.stderr
            )

            if is_download:
                DataRetriever._download(source=source, logger=logger)

        return data

    # noinspection PyUnusedFunction
    @staticmethod
    def does_exist(path):
        """
        Parameters
        ----------
        path : str
            path to requested resource (data or source file)

        Returns
        -------
        bool
            if the requested resource has either a data or source file available
        """
        return DataRetriever.has_data(path) or DataRetriever.has_source(path)

    @staticmethod
    def has_data(path):
        """
        Parameters
        ----------
        path : str
            path to requested resource (data or source file)

        Returns
        -------
        bool
            if the requested resource has a data file available
        """
        data_path = DataRetriever._get_data_path(path)
        is_file = os.path.isfile(data_path)

        if is_file and not os.path.getsize(data_path):
            # delete file with size of 0 bytes (probably from a failed download)
            os.remove(data_path)
            return False

        return is_file

    @staticmethod
    def has_source(path):
        """
        Parameters
        ----------
        path : str
            path to requested resource (data or source file)

        Returns
        -------
        bool
            if the requested resource has a source file available
        """
        return os.path.isfile(DataRetriever._get_source_path(path))

    @staticmethod
    def _get_data_path(path):
        if path.endswith(DataRetriever._EXTENSION):
            return path[: -len(DataRetriever._EXTENSION) - 1]
        else:
            return path

    @staticmethod
    def _get_source_path(path):
        if path.endswith(DataRetriever._EXTENSION):
            return path
        else:
            return f"{path}.{DataRetriever._EXTENSION}"

    @staticmethod
    def _download(source, logger=None):
        """
        Attempt to download data from the URL provided in the source file. Afterward, attempt to
        unpack in case the downloaded (or already available) file is an archive. Otherwise, show
        additional instructions in case such are provided in the source file.

        Parameters
        ----------
        source : str
            path to source file of requested resource
        logger : logging.logger, optional
            instance to provide identical logging behaviour as the calling process
        """

        from urllib import request
        from urllib.error import URLError
        import shutil
        from zipfile import is_zipfile

        # gather source file content
        log_str = f'opening file "{os.path.relpath(source)}"'
        try:
            with open(file=source, mode="r") as file:
                source_info = file.read().splitlines()
        except IOError:
            raise ValueError(f"{log_str}\n --> file not accessible")

        # strip spaces and remove empty lines
        source_info = [line.strip() for line in source_info if line]

        # gather download file path
        if source_info[0].upper().endswith(".ZIP"):
            # zip archive will be downloaded and unpacked
            download = os.path.join(
                os.path.dirname(source), os.path.basename(source_info[0])
            )
        else:
            # file will be downloaded directly
            download = DataRetriever._get_data_path(source)

        if os.path.isfile(download):
            # download was performed before
            log_str = (
                f'{log_str}\n --> data from URL "{source_info[0]}" already exists ...'
            )
        else:
            # execute download
            log_str = (
                f'{log_str}\n --> downloading data from URL "{source_info[0]}" ...'
            )
            try:
                with request.urlopen(source_info[0]) as response, open(
                    file=download, mode="wb"
                ) as file:
                    if response.length is None:
                        response.length = 0
                    log_str = f"{log_str}\n --> size: {response.length / 1e6:.1f} MB"
                    logger.warning(log_str) if logger else print(
                        f"[WARNING]  {log_str}", file=sys.stderr
                    )
                    sleep(0.05)  # to get correct output order

                    log_str = (
                        f"... download finished\n --> saving data into "
                        f'"{os.path.relpath(download)}"'
                    )
                    shutil.copyfileobj(fsrc=response, fdst=file)
            except URLError:
                raise ValueError(f"{log_str}\n --> URL not accessible")

        if len(source_info) > 1 and not is_zipfile(download):
            # gather further source instructions
            source_info = "\n     " + "\n     ".join(source_info[1:])
            log_str = f"{log_str}\n --> further instructions:{source_info}"

        # log information
        logger.warning(log_str) if logger else print(
            f"[WARNING]  {log_str}", file=sys.stderr
        )
        sleep(0.05)  # to get correct output order

        if len(source_info) > 1 and is_zipfile(download):
            DataRetriever._unpack(
                archive_file=download,
                member_file=source_info[1],
                target_file=source,
                logger=logger,
            )

    @staticmethod
    def _unpack(archive_file, member_file, target_file, logger=None):
        """
        Attempt to unpack a downloaded (or already available) archive file. Currently, only *.zip
        files are tested. Look at the provided exemplary source files for the supported syntax!

        Parameters
        ----------
        archive_file : str
            path to archive file of requested resource
        member_file : str
            path to data file withing archive of requested resource
        target_file : str
            path to data file destination of requested resource
        logger : logging.logger, optional
            instance to provide identical logging behaviour as the calling process
        """

        import shutil
        from zipfile import ZipFile

        # make sure target is a data and not a source file
        target_file = DataRetriever._get_data_path(target_file)
        if os.path.isfile(target_file):
            raise ValueError(
                f' --> target file "{os.path.relpath(target_file)}" already exists'
            )

        log_str = f' --> unpacking archive "{os.path.relpath(archive_file)}" ...'
        try:
            with ZipFile(file=archive_file, mode="r") as file:
                log_str = f'{log_str}\n --> unpacking member "{member_file}"'
                member = file.open(name=member_file, mode="r")

                # copy file (taken from ZipFile's extract)
                log_str = (
                    f'{log_str}\n --> saving data into "{os.path.relpath(target_file)}"'
                )
                target = open(file=target_file, mode="wb")
                with member, target:
                    shutil.copyfileobj(fsrc=member, fdst=target)

            # log information
            logger.warning(log_str) if logger else print(
                f"[WARNING]  {log_str}", file=sys.stderr
            )
            sleep(0.05)  # to get correct output order
        except IOError:
            raise ValueError(f"{log_str}\n --> file not accessible")
        except KeyError:
            raise ValueError(
                f"{log_str}\n --> member not found, contents are:\n{file.namelist()}"
            )
