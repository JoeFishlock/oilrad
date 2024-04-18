"""Module to calculate the optical properties for ice and ice containing oil droplets"""

import numpy as np

"""Load data for imaginary refractive index against wavelength from
doi:10.1029/2007JD009744.

To interpolate the data to other wavelengths should interpolate the log of the data
linearly.
"""
WARREN_DATA = np.loadtxt("oilrad/data/Warren_2008_ice_refractive_index.dat")
WARREN_WAVELENGTHS = WARREN_DATA[:, 0]  # in microns
WARREN_IMAGINARY_REFRACTIVE_INDEX = WARREN_DATA[:, 2]  # dimensionless

SCATTERING_COEFFICIENT_PEROVICH_1990_WHITE_ICE_INTERIOR = 2.5  # in 1/m

#################################
#  Pure ice optical properties  #
#################################


def calculate_ice_imaginary_refractive_index(wavelength):
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
    imaginary_refractive_index = calculate_ice_imaginary_refractive_index(
        wavelength_in_nm * 1e-3
    )
    absorption_coefficient = 4 * np.pi * imaginary_refractive_index / wavelengths_in_m
    return absorption_coefficient


def calculate_ice_scattering_coefficient_from_Roche_2022(ice_type: str):
    """Calculate ice scattering coefficient (1/m)
    doesn't depend on wavelength
    """
    ICE_ASYMMETRY_PARAM_ROCHE_2022 = 0.98  # dimensionless
    ICE_DENSITY_ROCHE_2022 = 800  # in kg/m3

    # mass cross section in m2/kg
    SCATTERING_MASS_CROSS_SECTION_ROCHE_2022 = {"FYI": 0.15, "MYI": 0.75, "MELT": 0.03}

    return (
        0.5
        * (1 - ICE_ASYMMETRY_PARAM_ROCHE_2022)
        * ICE_DENSITY_ROCHE_2022
        * SCATTERING_MASS_CROSS_SECTION_ROCHE_2022[ice_type]
    )


def calculate_ice_extinction_coefficient(wavelength_in_nm, ice_type):
    k = calculate_ice_absorption_coefficient(wavelength_in_nm)
    r = calculate_ice_scattering_coefficient_from_Roche_2022(ice_type)
    return np.sqrt(k**2 + 2 * k * r)


############################
#  oil optical properties  #
############################


def calculate_pure_oil_absorption_coefficient(wavelengths_in_nm):
    """Linearly interpolate data for pure Romashkino oil from Otremba 2007.
    This neflects the mie calculation droplet size effects.
    """
    DATA_WAVELENGTHS = np.array([350, 400, 450, 500, 550, 600, 650, 700, 750])  # in nm
    OIL_ABSORPTION = np.array(
        [215420, 20110, 7260, 3270, 1830, 1260, 770, 540, 340]
    )  # in 1/m
    return np.interp(wavelengths_in_nm, DATA_WAVELENGTHS, OIL_ABSORPTION)


def calculate_ice_oil_absorption_coefficient(wavelengths_in_nm, oil_mass_ratio):
    """Approximate the absorption coefficient of ice containing oil mass ratio
    in 1/m as pure ice absorption with mass ratio of pure oil.

    mass ratio in units of ng oil / g ice

    This is temporary until we can use better data
    """
    mass_ratio_dimensionless = oil_mass_ratio * 1e-9
    return calculate_ice_absorption_coefficient(
        wavelengths_in_nm
    ) + mass_ratio_dimensionless * calculate_pure_oil_absorption_coefficient(
        wavelengths_in_nm
    )


def calculate_ice_oil_extinction_coefficient(
    wavelength_in_nm, oil_mass_ratio, ice_type: str
):
    """oil mass ratio in ng oil/g ice yields extincition coefficient with oil pollution
    in 1/m
    """
    k = calculate_ice_oil_absorption_coefficient(wavelength_in_nm, oil_mass_ratio)
    r = calculate_ice_scattering_coefficient_from_Roche_2022(ice_type)
    return np.sqrt(k**2 + 2 * k * r)
