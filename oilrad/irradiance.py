"""Classes to store solution of two stream spectral model and integrate over
a given incident shortwave spectrum to return spectrally integrated properties of the solution.
"""

from dataclasses import dataclass
from numpy.typing import NDArray
from scipy.integrate import trapezoid
from .spectra import BlackBodySpectrum


@dataclass(frozen=True)
class SpectralIrradiance:
    """Two dimensional arrays containing the upwelling and downwelling irradiances at each
    depth and wavelength.

    Irradiances are non-dimensional and need to be multiplied by the incident spectral radiation.

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
        """Calculate spectral net irradiance"""
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
class Irradiance:
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


def integrate_over_SW(
    spectral_irradiance: SpectralIrradiance, spectrum: BlackBodySpectrum
) -> Irradiance:
    """Integrate over the spectral two-stream model solution over a given incident
    shortwave spectrum

    Args:
        spectral_irradiance (SpectralIrradiance): spectral two-stream model solution
        spectrum (BlackBodySpectrum): incident shortwave spectrum
    Returns:
        Irradiance: spectrally integrated irradiances
    """
    wavelengths = spectral_irradiance.wavelengths
    integrate = lambda irradiance: trapezoid(
        irradiance * spectrum(wavelengths), wavelengths, axis=1
    )
    integrated_upwelling = integrate(spectral_irradiance.upwelling)
    integrated_downwelling = integrate(spectral_irradiance.downwelling)
    return Irradiance(
        spectral_irradiance.z,
        integrated_upwelling,
        integrated_downwelling,
        _ice_base_index=spectral_irradiance._ice_base_index,
    )
