"""Classes to store solution of two stream spectral model and integrate over
a given incident shortwave spectrum to return spectrally integrated properties of the solution.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import trapezoid

from oilrad.constants import (
    PLANCK,
    LIGHTSPEED,
    MOLE,
    CLOUDY_SKY_FRACTIONS,
    WAVELENGTH_BAND_INDICES,
    calculate_band_snow_albedo,
    calculate_band_SSL_albedo,
    calculate_band_snow_transmittance,
    calculate_band_SSL_transmittance,
)
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
class SixBandSpectralIrradiance:
    """Two dimensional arrays containing the upwelling and downwelling irradiances at each
    depth and in each wavelength band.

    Irradiances are non-dimensional and need to be multiplied by the incident spectral radiation.

    Args:
        z (NDArray): vertical grid specified in dimensional units (m)
        upwelling (NDArray): 2D array of upwelling irradiances
        downwelling (NDArray): 2D array of downwelling irradiances
        snow_depth (float): snow depth in meters
        SSL_depth (float): SSL depth in meters
    """

    z: NDArray
    upwelling: NDArray
    downwelling: NDArray
    snow_depth: float
    SSL_depth: float

    _ice_base_index: int = 0

    @property
    def net_irradiance(self) -> NDArray:
        """Calculate spectral net irradiance"""
        return self.downwelling - self.upwelling

    @property
    def albedo(self) -> NDArray:
        """Calculate spectral albedo in each wavelength band

        This includes the albedo of the snow and the SSL above the ice."""
        ice_albedo = self.upwelling[-1, :]
        snow_albedo = np.array(
            [
                calculate_band_snow_albedo(self.snow_depth, i)
                for i in WAVELENGTH_BAND_INDICES
            ]
        )
        SSL_albedo = np.array(
            [
                calculate_band_SSL_albedo(self.SSL_depth, i)
                for i in WAVELENGTH_BAND_INDICES
            ]
        )
        snow_transmittance = np.array(
            [
                calculate_band_snow_transmittance(self.snow_depth, i)
                for i in WAVELENGTH_BAND_INDICES
            ]
        )
        SSL_transmittance = np.array(
            [
                calculate_band_SSL_transmittance(self.SSL_depth, i)
                for i in WAVELENGTH_BAND_INDICES
            ]
        )
        albedo = np.empty_like(ice_albedo)
        for i in WAVELENGTH_BAND_INDICES:
            albedo[i] = (
                snow_albedo[i]
                + snow_transmittance[i] * SSL_albedo[i]
                + snow_transmittance[i] * SSL_transmittance[i] * ice_albedo[i]
            )
        return albedo

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


@dataclass(frozen=True)
class SixBandIrradiance:
    """One dimensional Arrays containing the upwelling and downwelling irradiances at each
    depth integrated over the wavelength bands.

    Irradiances are non-dimensional and need to be multiplied by the incident spectral radiation.

    Args:
        z (NDArray): vertical grid specified in dimensional units (m)
        upwelling (NDArray): 1D array of integrated upwelling irradiances
        downwelling (NDArray): 1D array of integrated downwelling irradiances
        albedo (float): spectrall integrated albedo.
    """

    z: NDArray
    upwelling: NDArray
    downwelling: NDArray
    albedo: float

    _ice_base_index: int = 0

    @property
    def net_irradiance(self) -> NDArray:
        """Calculate net irradiance"""
        return self.downwelling - self.upwelling

    @property
    def transmittance(self) -> NDArray:
        """Calculate transmittance at the ice ocean interface or the bottom
        of the domain if the domain is entirely ice."""
        return self.downwelling[self._ice_base_index]


def integrate_over_SW(
    spectral_irradiance: SpectralIrradiance | SixBandSpectralIrradiance,
    spectrum: Optional[BlackBodySpectrum] = None,
) -> Irradiance | SixBandIrradiance:
    """Integrate over the spectral two-stream model solution over a given incident
    shortwave spectrum.

    When integrating the infinite layer model solution you must specify a normalised
    spectrum over which to integrate.

    Inegration over the six band solution is done using the fraction of incident
    radiaiton in each band under cloudy sky conditions (defined in oilrad.constants).

    Args:
        spectral_irradiance (SpectralIrradiance | SixBandSpectralIrradiance): spectral two-stream model solution
        spectrum (Optional[BlackBodySpectrum]): normalised incident shortwave spectrum (only needed when integrating SpectralIrradiance).
    Returns:
        Irradiance | SixBandIrradiance: spectrally integrated irradiances
    """
    if isinstance(spectral_irradiance, SpectralIrradiance):
        if spectrum is None:
            raise ValueError(
                "spectrum must be provided to integrate SpectralIrradiance"
            )
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
    elif isinstance(spectral_irradiance, SixBandSpectralIrradiance):
        integrate = lambda x: np.sum(x * CLOUDY_SKY_FRACTIONS, axis=1)
        return SixBandIrradiance(
            spectral_irradiance.z,
            integrate(spectral_irradiance.upwelling),
            integrate(spectral_irradiance.downwelling),
            np.sum(CLOUDY_SKY_FRACTIONS * spectral_irradiance.albedo),
            _ice_base_index=spectral_irradiance._ice_base_index,
        )
    else:
        raise NotImplementedError()
