#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
(c) 2023 under a MIT License (https://mit-license.org)

Authors:
- Lukas Brunner || l.brunner@univie.ac.at

Abstract: Time logging functions (copied)
"""
import logging
import traceback
from datetime import datetime

logger = logging.getLogger(__name__)
format_ = '%(asctime)s - %(levelname)s - %(funcName)s() %(lineno)s: %(message)s'


def set_logger(level='info', filename=None, format_=format_, **kwargs):
    """Set up a basic logger"""
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    logging.basicConfig(
        level=level,
        filename=filename,
        format=format_,
        **kwargs)
    

class LogTime:
    """A logger for keeping track of code timing.

    If called by 'with' log 'msg' with given 'level' before the intended code
    is executed. Log 'msg' again afterwards and add the status ('DONE' or
    'FAIL') as well as the execution time.

    The logger can be manually started and stopped by calling self.start() and
    self.stop (see examples).

    Parameters
    ----------
    msg : str, optional
        The default message to log.
    level : {'debug', 'info', 'warning'}, optional
        The default logging level.

    Examples
    --------
    with LogRegion('code description', level='info'):
        pass
    >>> INFO:code description...
    >>> INFO:code description... DONE (duration: 00:00:00.000000)

    with LogRegion('code description', level='info'):
        raise ValueError('error')
    >>> INFO:code description...
    >>> ERROR:code description... FAIL (duration: 00:00:00.000000)
    >>> ERROR:<Traceback>

    log = LogRegion('default message')
    log.start(level='debug')
    # calling start on a running logger will end the previous logger first
    log.start('other message')  # level will fall back to default
    log.stop
    >>> DEBUG:default message...
    >>> DEBUG:default message... DONE (duration: 00:00:00.000000)
    >>> INFO:other message...
    >>> INFO:other message... DONE (duration: 00:00:00.000000)

    with LogRegion('default message') as log:
        log.stop  # explicitly stop previous logger (optional)
        # piece of code which is not timed here
        log.start('other message', level='debug')
        log.start()  # fall back to defaults
        raise ValueError('error message')
    >>> INFO:default message...
    >>> INFO:default message... DONE (duration: 00:00:00.000000)
    >>> DEBUG:other message...
    >>> DEBUG:other message... DONE (duration: 00:00:00.000000)
    >>> INFO:default message...
    >>> ERROR:default message... FAIL (duration: 00:00:00.000000)
    >>> ERROR:<Traceback>
    """

    def __init__(self, msg='Start logging', level='info'):
        self.default_msg = msg
        self.default_level = level
        self.running = False

    def __enter__(self):
        self.start(self.default_msg, self.default_level)
        return self

    def __exit__(self, exception_type, exception_value, tb):
        if exception_type is None:
            self.stop
        else:
            self.level = 'error'
            self.stop
            # self.log_region(f'{exception_type.__name__}: {exception_value}')
            self.log_region(''.join(traceback.format_exception(
                exception_type, exception_value, tb)))

    def start(self, msg=None, level=None):
        """Log msg with given level.

        If LogTime is already running (because self.start() has already been
        called without a subsequent self.stop) this will also call self.stop
        before any other action (see self.stop for more information)

        Parameters
        ----------
        msg : str, optional
            Message to log.
        level : {'debug', 'info', 'warning'}, optional
            Overwrite class logging level for this call.
        """
        if self.running:
            self.stop
        self.running = True

        if msg is None:
            self.msg = self.default_msg
        else:
            self.msg = msg

        if level is None:
            self.level = self.default_level
        else:
            self.level = level

        self.t0 = datetime.now()
        self.log_region(f'{self.msg}...')

    @property
    def stop(self):
        """Log msg (from self.start) again and indicate time passed."""
        try:
            dt = datetime.now() - self.t0
        except AttributeError:
            raise ValueError('Timer not running, call self.start() first')
        if self.level == 'error':
            self.log_region(f'{self.msg}... FAIL (duration: {dt})')
        else:
            self.log_region(f'{self.msg}... DONE (duration: {dt})')
        self.running = False

    def log_region(self, msg):
        # this only exists to that the logger prints a nice function name
        self._logger(self.level)(msg)

    @staticmethod
    def _logger(level):
        """Set logging level from string"""
        if level.lower() == 'debug':
            return logger.debug
        elif level.lower() == 'info':
            return logger.info
        elif level.lower() == 'warning':
            return logger.warning
        elif level.lower() == 'error':
            return logger.error