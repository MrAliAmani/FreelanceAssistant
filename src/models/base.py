from abc import ABC, abstractmethod
from typing import List, Optional, Union

class BaseModel(ABC):
    """Base class for all model interfaces"""
    
    @abstractmethod
    def test_connection(self) -> bool:
        """Test if the model connection is working"""
        pass 