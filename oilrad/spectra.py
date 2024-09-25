from typing import Callable
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad

"""Data from:
https://www.oceanopticsbook.info/view/light-and-radiometry/level-2/light-from-the-sun
"""


PLANCK = 6.62607015e-34  # Js
LIGHTSPEED = 299792458  # m/s
BOLTZMANN = 1.380649e-23  # J/K
AU = 1.496e11  # m
SUN_RADIUS = 6.95e8  # m


PLANCK_FUNCTION = lambda L, T: (2 * PLANCK * LIGHTSPEED**2 / L**5) * (
    1 / (np.exp(PLANCK * LIGHTSPEED / (BOLTZMANN * L * T)) - 1)
)


def top_of_atmosphere_irradiance(wavelength):
    """For wavelength in nm and temperature in K return top of atmosphere solar
    irradiance in W/m2 nm
    https://www.oceanopticsbook.info/view/light-and-radiometry/level-2/blackbody-radiation
    """
    wavelength_in_m = wavelength * 1e-9
    return (
        PLANCK_FUNCTION(wavelength_in_m, T=5782)
        * (SUN_RADIUS**2 / AU**2)
        * np.pi
        * 1e-9
    )


def get_normalised_black_body_spectrum(
    wavelength_range: tuple[float, float]
) -> Callable[[NDArray], NDArray]:
    """Return a solar black body spectrum which is normalised over the wavelength range"""
    max_wavelength = wavelength_range[1]
    min_wavelength = wavelength_range[0]
    TOTAL_IRRADIANCE = quad(
        top_of_atmosphere_irradiance, min_wavelength, max_wavelength
    )[0]

    def spectrum(wavelength_in_nm: NDArray) -> NDArray:
        if np.any(wavelength_in_nm > max_wavelength) or np.any(
            wavelength_in_nm < min_wavelength
        ):
            raise ValueError(
                f"wavelength not in shortwave range {min_wavelength}nm - {max_wavelength}nm"
            )
        return top_of_atmosphere_irradiance(wavelength_in_nm) / TOTAL_IRRADIANCE

    return spectrum
