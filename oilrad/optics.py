"""Module to calculate the optical properties for ice containing oil droplets:
- The spectral absorption coefficient which depends on oil mass concentration and droplet size
- The wavelength-independent scattering coefficient which takes a constant value in the ice and is zero the liquid

Load data for imaginary refractive index against wavelength from
doi:10.1029/2007JD009744.
This is used to calculate the absorption coefficient of pure ice.
To interpolate the data to other wavelengths should interpolate the log of the data
linearly.

Oil absorption calculated following Redmond Roche et al 2022 using given data for mass
absorption coefficient of oil in ice which is a function of wavelength and droplet radius
"""

from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import LinearNDInterpolator
from .constants import ICE_DENSITY_ROCHE_2022

# Load Warrent data for pure ice absorption
DATADIR = Path(__file__).parent / "data"
WARREN_DATA = np.loadtxt(DATADIR / "Warren_2008_ice_refractive_index.dat")
WARREN_WAVELENGTHS = WARREN_DATA[:, 0]  # in microns
WARREN_IMAGINARY_REFRACTIVE_INDEX = WARREN_DATA[:, 2]  # dimensionless


# Create a 2D array of droplet sizes and wavelengths we can interpolate for MAC
# The wavelengths are always the same so only need to read once
ROMASHKINO_DROPLET_RADII = np.array([0.05, 0.25, 0.5, 1.5, 2.5, 3.5, 5.0])
wavelengths = np.genfromtxt(
    DATADIR / "MassAbsCoe/Romashkino/MAC_0.05.dat",
    delimiter=",",
    skip_header=1,
)[:, 0]
OIL_MAC_DATA = np.empty((wavelengths.size, len(ROMASHKINO_DROPLET_RADII)))
for i, droplet_size in enumerate(ROMASHKINO_DROPLET_RADII):
    MACs = np.genfromtxt(
        DATADIR / f"MassAbsCoe/Romashkino/MAC_{droplet_size}.dat",
        delimiter=",",
        skip_header=1,
    )[:, 1]
    OIL_MAC_DATA[:, i] = MACs

# Set up interpolator for oil MAC data
long_wavelengths = np.tile(wavelengths, len(ROMASHKINO_DROPLET_RADII))
long_radii = np.repeat(ROMASHKINO_DROPLET_RADII, len(wavelengths))
interp = LinearNDInterpolator(
    list(zip(long_wavelengths, long_radii)), OIL_MAC_DATA.flatten("F"), rescale=True
)

#################################
#  Pure ice optical properties  #
#################################


def _calculate_ice_imaginary_refractive_index(wavelength):
    """Interpolate warren data to return imaginary index for given wavelengths.

    wavelength array must be inputted in microns
    """
    interpolated_log_refractive_index = np.interp(
        np.log(wavelength),
        np.log(WARREN_WAVELENGTHS),
        np.log(WARREN_IMAGINARY_REFRACTIVE_INDEX),
    )
    return np.exp(interpolated_log_refractive_index)


def calculate_ice_absorption_coefficient(wavelength_in_nm):
    """calculate ice absorption coefficient from Warren 2008 data at given
    wavelengths inputted in nano meters from interpolated imaginary refractive index
    data"""
    wavelengths_in_m = wavelength_in_nm * 1e-9
    imaginary_refractive_index = _calculate_ice_imaginary_refractive_index(
        wavelength_in_nm * 1e-3
    )
    absorption_coefficient = 4 * np.pi * imaginary_refractive_index / wavelengths_in_m
    return absorption_coefficient


def calculate_scattering(liquid_fraction: NDArray, ice_scattering_coefficient: float):
    """Calculate scattering coefficient in ice and return zero in liquid.
    (doesn't depend on wavelength)

    Smoothly transitions from the ice_scattering_coefficient to zero in the liquid using a tanh function.

    Args:
        liquid_fraction (NDArray): liquid fraction as a function of depth
        ice_scattering_coefficient (float): scattering coefficient of ice

    Returns:
        NDArray: scattering coefficient [1/m] as a function of depth
    """

    return ice_scattering_coefficient * np.tanh((1 - liquid_fraction) * 100)


############################
#  oil optical properties  #
############################


def Romashkino_MAC(wavelength_nm, droplet_radius_microns):
    return np.where(
        wavelength_nm > 800, 0, interp(wavelength_nm, droplet_radius_microns)
    )


def calculate_ice_oil_absorption_coefficient(
    wavelengths_in_nm,
    oil_mass_ratio,
    droplet_radius_in_microns,
    absorption_enhancement_factor=1.0,
):
    """Calculate the absorption coefficient in 1/m of ice polluted with oil droplets
    following roche et al 2022. The oil droplets radii are distributed log-normally
    with geometric standard deviation e. We specify the median radius for the distribution.

    mass ratio in units of ng oil / g ice

    This is for Romashkino oil.

    The enahncement factor is an ad hoc correction for the two stream model to try and
    better match the results of redmondroche2022 which used an 8-stream model

    Args:
        wavelengths_in_nm (NDArray): wavelengths in nm
        oil_mass_ratio (float): mass ratio of oil in ice in ng/g
        droplet_radius_in_microns (float): median droplet radius in microns
        absorption_enhancement_factor (float, optional): enhancement factor for absorption coefficient. Defaults to 1.0.

    Returns:
        NDArray: absorption coefficient in 1/m
    """
    mass_ratio_dimensionless = oil_mass_ratio * 1e-9
    return absorption_enhancement_factor * (
        calculate_ice_absorption_coefficient(wavelengths_in_nm)
        + mass_ratio_dimensionless
        * 1e3
        * ICE_DENSITY_ROCHE_2022
        * Romashkino_MAC(wavelengths_in_nm, droplet_radius_in_microns)
    )
