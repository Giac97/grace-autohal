"""Foundation-model-driven active learning for GRACE fine-tuning."""

from importlib import metadata

try:
    __version__ = metadata.version("grace-al")
except metadata.PackageNotFoundError:  # running from a source tree
    __version__ = "0.0.0+unknown"
