from abc import ABC, abstractmethod

class ImageOperation(ABC):

    def __init__(self, **kwargs):
        self.parameters = kwargs

    @abstractmethod
    def validateParameters(self):
        pass

    def compute(self):
        self.validateParameters()
        return self._compute()

    @abstractmethod
    def _compute(self):
        pass
