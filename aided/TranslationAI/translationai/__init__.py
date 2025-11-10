import signal
try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    # Python < 3.8
    from importlib_metadata import version, PackageNotFoundError

signal.signal(signal.SIGINT, lambda x, y: exit(0))

name = 'translationai'
try:
    __version__ = version(name)
except PackageNotFoundError:
    # Package is not installed
    __version__ = "unknown"