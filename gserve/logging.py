import logging

LOG_FORMAT = "%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s"
LOG_DATEFMT = "%m-%d %H:%M:%S"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure basic logging for the library.

    If the root logger has no handlers configured, this sets up a simple
    configuration using :func:`logging.basicConfig` with the standard
    format and the given log level. Subsequent calls have no effect if
    handlers are already present.
    """
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=LOG_DATEFMT)
