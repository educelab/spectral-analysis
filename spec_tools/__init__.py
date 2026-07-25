from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version('spectral-analysis')
except PackageNotFoundError:
    # Running from a source tree that was never installed. The apps stamp this
    # into output metadata, so report it as unknown rather than silently
    # claiming a version we cannot substantiate.
    __version__ = '0.0.0+unknown'

__all__ = ['__version__']
