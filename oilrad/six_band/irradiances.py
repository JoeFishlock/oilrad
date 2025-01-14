"""Classes to store spectral and integrated solution of the six band model.
All irradiances are normalised by the incident.
"""

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from ..constants import (
    PLANCK,
    LIGHTSPEED,
    MOLE,
    CLOUDY_SKY_FRACTIONS,
)


@dataclass(frozen=True)
class SixBandSpectralIrradiance:
    """Two dimensional arrays containing the upwelling and downwelling irradiances at
    each depth and in each wavelength band.

    Irradiances are non-dimensional and need to be multiplied by the incident spectral
    radiation.

    Args:
        z (NDArray): vertical grid specified in dimensional units (m)
        upwelling (NDArray): 2D array of upwelling irradiances
        downwelling (NDArray): 2D array of downwelling irradiances
        albedo (NDArray): 1D array of spectral albedo (including snow and SSL)
    """

    z: NDArray
    upwelling: NDArray
    downwelling: NDArray
    albedo: NDArray

    _ice_base_index: int = 0

    @property
    def net_irradiance(self) -> NDArray:
        return self.downwelling - self.upwelling

    @property
    def transmittance(self) -> NDArray:
        """Calculate spectral transmittance at the ice ocean interface or the bottom
        of the domain if the domain is entirely ice."""
        return self.downwelling[self._ice_base_index, :]

    @property
    def PAR_transmittance(self) -> NDArray:
        """Calculate plane PAR transmittance as the ratio of the
        net irradiant power in the PAR range (400-700nm) to the incident
        irradiative power at the ice / snow surface.
        """
        return np.sum(
            self.net_irradiance[:, 1:4] * CLOUDY_SKY_FRACTIONS[1:4], axis=1
        ) / np.sum(CLOUDY_SKY_FRACTIONS[1:4])

    @property
    def plane_PAR(self) -> NDArray:
        """Calculate plane PAR normalised by the incident broadband shortwave irradiance.

        To convert to micromol-photns m^-2 s^-1 we need to multiply by the incident shortwave
        irradiance in W m^-2.

        To convert to the scalar value for an isotropic downwelling irradiance multiply
        by a factor of 2.
        """
        PAR_weightings = np.array(
            [9.809215701925024e-08, 1.0750608149135324e-07, 1.0006054876321231e-07]
        )
        return (1e6 / (PLANCK * LIGHTSPEED * MOLE)) * np.sum(
            (self.upwelling[:, 1:4] + self.downwelling[:, 1:4]) * PAR_weightings, axis=1
        )

    @property
    def ice_base_PAR_transmittance(self) -> float:
        return self.PAR_transmittance[self._ice_base_index]

    @property
    def ice_base_plane_PAR(self) -> float:
        return self.plane_PAR[self._ice_base_index]


@dataclass(frozen=True)
class SixBandIrradiance:
    """One dimensional Arrays containing the upwelling and downwelling irradiances at each
    depth integrated over the wavelength bands.

    Irradiances are non-dimensional and need to be multiplied by the incident spectral radiation.

    Args:
        z (NDArray): vertical grid specified in dimensional units (m)
        upwelling (NDArray): 1D array of integrated upwelling irradiances
        downwelling (NDArray): 1D array of integrated downwelling irradiances
        albedo (float): spectrally integrated albedo (including the snow layer and SSL)
    """

    z: NDArray
    upwelling: NDArray
    downwelling: NDArray
    albedo: float

    _ice_base_index: int = 0

    @property
    def net_irradiance(self) -> NDArray:
        return self.downwelling - self.upwelling

    @property
    def transmittance(self) -> NDArray:
        """Calculate transmittance at the ice ocean interface or the bottom
        of the domain if the domain is entirely ice."""
        return self.downwelling[self._ice_base_index]
