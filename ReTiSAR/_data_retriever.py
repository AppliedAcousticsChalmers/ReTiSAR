import os.path
import sys


class DataRetriever(object):
    """
    Flexible interface to check the availability of requested project resources. Since resources cannot be distributed
    as part of this publication, only links to publicly available resources can be provided. When a certain recourse
    is requested on startup of a rendering component, its availability or otherwise existence of a respective source
    reference is checked.
    """

    _EXTENSION = 'source'
    """File extension used for the reference files."""

    @staticmethod
    def retrieve(path, is_download=True, logger=None):
        """
        Check for existence of the requested resource. Depending on the provided file (data or source),
        different steps will be performed in order to allow the application to find the data file, or allow the user
        to give instruction to provide the data file.

        In the current implementation the data will be attempted to download in case no further instructions despite the
        download URL are provided in the source file. Otherwise, the reference link and further instructions are logged
        and shown for the user. The startup process will then be interrupted in case the data file is not available.

        Parameters
        ----------
        path : str or numpy.ndarray
            path to requested resource (data or source file) or directly provided filter coefficients where latter are
            returned directly
        is_download : bool, optional
            if file should be downloaded in case no data but a source file is available
        logger : logging.logger, optional
            instance to provide identical logging behaviour as the calling process

        Returns
        -------
        str or numpy.ndarray
            path to resource data file (not necessarily available) or filter coefficients in case they were directly
            provided
        """
        if not isinstance(path, str):
            return path

        data = DataRetriever._get_data_path(path)
        if DataRetriever.has_data(path):
            if data != path:
                log_str = f'source file "{os.path.relpath(path)}" given, but data is already available.'
                logger.warning(log_str) if logger else print(f'[WARNING]  {log_str}', file=sys.stderr)

        elif DataRetriever.has_source(path):
            source = DataRetriever._get_source_path(path)
            if path != source:
                log_str = f'not yet available data file "{os.path.relpath(path)}" given, but source file was found.'
            else:
                log_str = f'source file "{os.path.relpath(path)}" given, but data is not yet available.'
            logger.warning(log_str) if logger else print(f'[WARNING]  {log_str}', file=sys.stderr)

            # gather source file content
            log_str = f'opening file "{os.path.relpath(source)}"'
            try:
                with open(source, mode='r') as f:
                    source_info = f.read().splitlines()
            except IOError:
                raise ValueError(f'{log_str}\n --> file not accessible')

            # strip spaces and remove empty lines
            source_info = [line.strip() for line in source_info if line]

            if is_download and len(source_info) == 1:
                # download in case data file is directly accessible (no further instructions like unpacking of archive
                # or renaming of file is necessary)
                from urllib import request
                from urllib.error import URLError
                import shutil

                # execute download
                log_str = f'{log_str}\n --> downloading data from URL "{source_info[0]}" ...'
                logger.warning(log_str) if logger else print(f'[WARNING]  {log_str}', file=sys.stderr)
                try:
                    with request.urlopen(source_info[0]) as response, open(data, 'wb') as file:
                        log_str = f'... download finished\n --> saving data into "{os.path.relpath(data)}"'
                        shutil.copyfileobj(response, file)
                except URLError:
                    raise ValueError(f'URL "{source_info[0]}" not accessible')
                logger.warning(log_str) if logger else print(f'[WARNING]  {log_str}', file=sys.stderr)

            else:
                # print source URL
                log_str = f'{log_str}\n --> download data yourselves from: {source_info[0]}'

                # print further source instruction
                if len(source_info) > 1:
                    source_info = '\n     ' + '\n     '.join(source_info[1:])
                    log_str = f'{log_str}\n --> further instructions:{source_info}'
                logger.warning(log_str) if logger else print(f'[WARNING]  {log_str}', file=sys.stderr)

        return data

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
        return os.path.isfile(DataRetriever._get_data_path(path))

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
            return path[:-len(DataRetriever._EXTENSION) - 1]
        else:
            return path

    @staticmethod
    def _get_source_path(path):
        if path.endswith(DataRetriever._EXTENSION):
            return path
        else:
            return f'{path}.{DataRetriever._EXTENSION}'
