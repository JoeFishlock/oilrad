"""Classes to store the solution of the continuous wavelength two stream model.
Spectral irradiances and the integrated irradiances noralised by the incident.
"""

from dataclasses import dataclass
from numpy.typing import NDArray


@dataclass(frozen=True)
class CtsWavelengthSpectralIrradiance:
    """Two dimensional arrays containing the upwelling and downwelling irradiances at
    each depth and wavelength.

    Irradiances are non-dimensional and need to be multiplied by the incident spectral
    radiation.

    Args:
        z (NDArray): vertical grid specified in dimensional units (m)
        wavelengths (NDArray): array of wavelengths in nm
        upwelling (NDArray): 2D array of upwelling irradiances
        downwelling (NDArray): 2D array of downwelling irradiances
    """

    z: NDArray
    wavelengths: NDArray
    upwelling: NDArray
    downwelling: NDArray

    _ice_base_index: int = 0

    @property
    def net_irradiance(self) -> NDArray:
        return self.downwelling - self.upwelling

    @property
    def albedo(self) -> NDArray:
        """Calculate spectral albedo"""
        return self.upwelling[-1, :]

    @property
    def transmittance(self) -> NDArray:
        """Calculate spectral transmittance at the ice ocean interface or the bottom
        of the domain if the domain is entirely ice."""
        return self.downwelling[self._ice_base_index, :]


@dataclass(frozen=True)
class CtsWavelengthIrradiance:
    """One dimensional Arrays containing the upwelling and downwelling irradiances at each
    depth integrated over wavelength.

    Irradiances are non-dimensional and need to be multiplied by the incident spectral radiation.

    Args:
        z (NDArray): vertical grid specified in dimensional units (m)
        upwelling (NDArray): 1D array of integrated upwelling irradiances
        downwelling (NDArray): 1D array of integrated downwelling irradiances
    """

    z: NDArray
    upwelling: NDArray
    downwelling: NDArray

    _ice_base_index: int = 0

    @property
    def net_irradiance(self) -> NDArray:
        """Calculate net irradiance"""
        return self.downwelling - self.upwelling

    @property
    def albedo(self) -> NDArray:
        """Calculate albedo"""
        return self.upwelling[-1]

    @property
    def transmittance(self) -> NDArray:
        """Calculate transmittance at the ice ocean interface or the bottom
        of the domain if the domain is entirely ice."""
        return self.downwelling[self._ice_base_index]
