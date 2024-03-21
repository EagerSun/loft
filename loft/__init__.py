import os

__version__ = "0.0.3"
__author__ = 'Eager Sun'
__credits__ = 'MIT'

def get_framework():
    return os.environ.get("PADDLE_OR_TORCH", "paddle")

def get_numgpu_per_node():
    return int(os.environ.get("CGPU_COUNT", "8"))

from .lofter import Lofter
from .utils import BaseMethod, BaseModel
from .utils import BaseRunner, Logger
