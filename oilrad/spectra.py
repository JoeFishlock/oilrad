"""Module to provide the spectrum of incident downwelling shortwave radiation at the
top of the domain.

Currently only the black body spectrum is integrated which follows the solar spectrum
at the top of the atmosphere, but is normalised to integrate to one in the shortwave range.

Data used to compute the Planck function is from:
https://www.oceanopticsbook.info/view/light-and-radiometry/level-2/light-from-the-sun
"""
from functools import cached_property
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad

from .constants import PLANCK, LIGHTSPEED, BOLTZMANN, AU, SUN_RADIUS


_solar_planck_function = lambda L, T: (2 * PLANCK * LIGHTSPEED**2 / L**5) * (
    1 / (np.exp(PLANCK * LIGHTSPEED / (BOLTZMANN * L * T)) - 1)
)


@dataclass(frozen=True)
class BlackBodySpectrum:
    """Spectrum with blackbody shape that integrates to 1 between minimum and maximum
    wavelength specified in nm

    Once initialised the spectrum can be called with an array of wavelengths in nm.

    Args:
        min_wavelength: minimum wavelength in nm
        max_wavelength: maximum wavelength in nm

    Raises:
        ValueError: if wavelength is not in the shortwave range
    """

    min_wavelength: float
    max_wavelength: float

    @cached_property
    def _total_irradiance(self) -> float:
        return quad(
            self._top_of_atmosphere_irradiance, self.min_wavelength, self.max_wavelength
        )[0]

    @classmethod
    def _top_of_atmosphere_irradiance(cls, wavelength_in_nm):
        """For wavelength in nm and temperature in K return top of atmosphere solar
        irradiance in W/m2 nm
        https://www.oceanopticsbook.info/view/light-and-radiometry/level-2/blackbody-radiation
        """
        return (
            _solar_planck_function(wavelength_in_nm * 1e-9, T=5782)
            * (SUN_RADIUS**2 / AU**2)
            * np.pi
            * 1e-9
        )

    def __call__(self, wavelength_in_nm: NDArray) -> NDArray:
        if np.any(wavelength_in_nm > self.max_wavelength) or np.any(
            wavelength_in_nm < self.min_wavelength
        ):
            raise ValueError(
                f"wavelength not in shortwave range {self.min_wavelength}nm - {self.max_wavelength}nm"
            )
        return (
            self._top_of_atmosphere_irradiance(wavelength_in_nm)
            / self._total_irradiance
        )
