import logging
import os
import sys

_FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s"
_DATE_FORMAT = "%m-%d %H:%M:%S"


class NewLineFormatter(logging.Formatter):
    """Adds logging prefix to newlines to align multi-line messages."""

    def __init__(self, fmt, datefmt=None):
        logging.Formatter.__init__(self, fmt, datefmt)

    def format(self, record):
        msg = logging.Formatter.format(self, record)
        if record.message != "":
            parts = msg.split(record.message)
            msg = msg.replace("\n", "\r\n" + parts[0])
        return msg


_root_logger = logging.getLogger("flash")
_root_logger.setLevel(logging.DEBUG)
_default_handler = None


def _setup_logger():
    _root_logger.setLevel(logging.DEBUG)
    global _default_handler

    _default_handler = logging.StreamHandler(sys.stdout)
    _default_handler.flush = sys.stdout.flush  # type: ignore
    _default_handler.setLevel(logging.DEBUG)
    _root_logger.addHandler(_default_handler)

    fmt = NewLineFormatter(_FORMAT, datefmt=_DATE_FORMAT)
    _default_handler.setFormatter(fmt)
    # Setting this will avoid the message
    # being propagated to the parent logger.
    _root_logger.propagate = False


_setup_logger()


def init_logger(name: str):
    # Use the same settings as above for root logger
    logger = logging.getLogger(name)

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()  # Changed default to INFO
    logger.setLevel(log_level)

    # Ensure the handler's level matches the logger's level
    if _default_handler:
        _default_handler.setLevel(log_level)

    logger.addHandler(_default_handler)
    logger.propagate = False

    return logger


def set_log_level(level: str):
    """
    Set the log level for all FlashEngine loggers.

    Args:
        level: Log level string ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    """
    level = level.upper()
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    if level not in valid_levels:
        raise ValueError(f"Invalid log level: {level}. Must be one of {valid_levels}")

    os.environ["LOG_LEVEL"] = level

    _root_logger.setLevel(level)
    if _default_handler:
        _default_handler.setLevel(level)

    for logger_name in logging.Logger.manager.loggerDict:
        if logger_name.startswith("flash"):
            existing_logger = logging.getLogger(logger_name)
            existing_logger.setLevel(level)


def get_log_level() -> str:
    """Get the current log level."""
    return os.getenv("LOG_LEVEL", "INFO").upper()


def disable_debug_logs():
    """Convenience function to only show WARNING, ERROR, and CRITICAL logs."""
    set_log_level("WARNING")


def enable_debug_logs():
    """Convenience function to show all log levels including DEBUG."""
    set_log_level("DEBUG")


def enable_info_logs():
    """Convenience function to show INFO and above (default)."""
    set_log_level("INFO")


def enable_minimal_logs():
    """Show only ERROR and CRITICAL logs."""
    set_log_level("ERROR")


def enable_quiet_mode():
    """Alias for minimal logs - show only errors."""
    enable_minimal_logs()
