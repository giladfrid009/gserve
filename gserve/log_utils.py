"""Utilities for configuring logging."""

from __future__ import annotations

import os
import logging


LOG_FORMAT = "%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s"
LOG_DATEFMT = "%m-%d %H:%M:%S"

#: Environment variable used to propagate the log level to subprocesses.
LOG_LEVEL_ENV = "GSERVE_LOG_LEVEL"


def setup_logging(level: int | None = None) -> None:
    """Initialize or update the root logger configuration.

    If a log handler already exists, its level and formatter are updated;
    otherwise a :class:`logging.StreamHandler` is installed.  The log level is
    also stored in :data:`LOG_LEVEL_ENV` so subprocesses started by the library
    inherit the same verbosity.
    """
    default_level = logging.INFO

    if level is None:
        # If caller didn't specify a level, try the environment variable.
        env_level = os.environ.get(LOG_LEVEL_ENV, default=logging.getLevelName(default_level))
        level = getattr(logging, env_level.upper(), None)

        # warning if env_level is not a valid logging level
        if level is None:
            level = default_level
            logging.warning(f"Invalid log level '{env_level}' specified in {LOG_LEVEL_ENV}. Defaulting to INFO.")

    root = logging.getLogger()
    root.setLevel(level)

    fmt = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT)

    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(fmt)
        handler.setLevel(level)
        root.addHandler(handler)
    else:
        for handler in root.handlers:
            handler.setFormatter(fmt)
            handler.setLevel(level)

    os.environ[LOG_LEVEL_ENV] = logging.getLevelName(level)
