"""Define an interface here that all solutions of the two stream radiation model
implement. Specifically once a model is initialised from its required parameters it
will provide methods to determine the upwelling radiation, downwelling radiation and
the radiative heating as functions of depth and wavelength. It will also provide
methods for the spectral albedo and transmission."""

from abc import ABC, abstractmethod


class AbstractModel(ABC):
    @abstractmethod
    def upwelling(self, z, L):
        pass

    @abstractmethod
    def downwelling(self, z, L):
        pass

    @abstractmethod
    def heating(self, z, L):
        pass

    @abstractmethod
    def albedo(self, L):
        pass

    @abstractmethod
    def transmittance(self, L):
        pass
