from abc import ABC
from typing import List
from layers.base import BaseLayer

class BaseNetwork(ABC):
    """Base Network class."""
    
    layers: List[BaseLayer]

    def __init__(self, layers: List[BaseLayer]):
        """Initialize the network by providing a list of layers"""
        self.layers = layers
