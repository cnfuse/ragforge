"""Structured, dependency-free logging setup.

A single :func:`configure` call wires up a consistent format across the CLI,
the API service, and library code. :func:`get_logger` is the accessor everything
else should use.
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def configure(level: str = "INFO") -> None:
    """Initialise root logging once with a readable single-line format."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, configuring logging lazily on first use."""
    if not _CONFIGURED:
        configure()
    return logging.getLogger(f"ragforge.{name}")
