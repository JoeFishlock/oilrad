"""Solution of two stream radiative transfer model in single ice layer
containing uniform oil mass concentration

Assume no Fresnel reflection

parameters:
oil_mass_ratio (ng oil/ g ice)
ice_thickness (m)
scattering and absorption coefficients (1/m) from optics module

"""

import numpy as np
from gulf.optics import (
    calculate_ice_oil_absorption_coefficient,
    calculate_ice_oil_extinction_coefficient,
    calculate_ice_scattering_coefficient_from_Roche_2022,
)

########################
#  Albedo calculation  #
########################


def calculate_albedo(ice_thickness, wavelength_in_nm, oil_mass_ratio, ice_type: str):
    """calculate spectral alebdo with no Fresnel reflection
    wavelength in nm
    ice thickness (m)
    oil mass ratio (ng oil/g ice)
    ice_type for scattering coefficient
    """
    total_absorption = calculate_ice_oil_absorption_coefficient(
        wavelength_in_nm, oil_mass_ratio
    )
    extinction_coefficient = calculate_ice_oil_extinction_coefficient(
        wavelength_in_nm, oil_mass_ratio, ice_type
    )
    scattering = calculate_ice_scattering_coefficient_from_Roche_2022(ice_type)
    optical_depth = extinction_coefficient * ice_thickness
    return scattering / (
        total_absorption
        + scattering
        + (extinction_coefficient / np.tanh(optical_depth))
    )


def calculate_optically_thick_albedo(wavelength_in_nm, oil_mass_ratio, ice_type: str):
    """calculate spectral alebdo for wavelength in nm in optically thick limit
    oil mass ratio (ng oil/g ice)
    ice_type for scattering coefficient
    """
    total_absorption = calculate_ice_oil_absorption_coefficient(
        wavelength_in_nm, oil_mass_ratio
    )
    extinction_coefficient = calculate_ice_oil_extinction_coefficient(
        wavelength_in_nm, oil_mass_ratio, ice_type
    )
    return (extinction_coefficient - total_absorption) / (
        extinction_coefficient + total_absorption
    )


#######################################
#  Upwelling / downwelling radiation  #
#######################################

"""
For computational stability try represnting single layer solution in exponential
basis:

upwelling = A exp(+ext z) + B exp(-ext z)
downwelling = C exp(+ext z) + D exp(-ext z)

B = -A exp(-2 optical depth)
C = A/s
D = 1 - A/s

s is optically thick single layer albedo
optical depth = ice_thickness * extinction coefficient

Non dimensionalise by the incident shortwave at each wavelength so that
upwelling radiation -> upwelling * incident shortwave spectrum
"""


def calculate_A(ice_thickness, wavelength_in_nm, oil_mass_ratio, ice_type: str):
    k = calculate_ice_oil_absorption_coefficient(wavelength_in_nm, oil_mass_ratio)
    mu = calculate_ice_oil_extinction_coefficient(
        wavelength_in_nm, oil_mass_ratio, ice_type
    )
    r = calculate_ice_scattering_coefficient_from_Roche_2022(ice_type)
    optical_depth = mu * ice_thickness
    s = (mu - k) / (mu + k)

    return s * (1 / (1 - np.exp(-2 * optical_depth) * ((k + r - mu) / (k + r + mu))))


def calculate_ND_upwelling(
    z, ice_thickness, wavelength_in_nm, oil_mass_ratio, ice_type: str
):
    """Calculate upwelling radiation dvided by shortwave incidence in ice
    -ice_thickness < z < 0
    """
    A = calculate_A(ice_thickness, wavelength_in_nm, oil_mass_ratio, ice_type)
    mu = calculate_ice_oil_extinction_coefficient(
        wavelength_in_nm, oil_mass_ratio, ice_type
    )
    optical_depth = mu * ice_thickness
    return A * (np.exp(mu * z) - np.exp(-mu * z) / np.exp(2 * optical_depth))


def calculate_ND_downwelling(
    z, ice_thickness, wavelength_in_nm, oil_mass_ratio, ice_type: str
):
    """Calculate downwelling radiation dvided by shortwave incidence in ice
    -ice_thickness < z < 0
    """
    A = calculate_A(ice_thickness, wavelength_in_nm, oil_mass_ratio, ice_type)
    mu = calculate_ice_oil_extinction_coefficient(
        wavelength_in_nm, oil_mass_ratio, ice_type
    )
    k = calculate_ice_oil_absorption_coefficient(wavelength_in_nm, oil_mass_ratio)
    s = (mu - k) / (mu + k)

    return (A / s) * np.exp(mu * z) + (1 - (A / s)) * np.exp(-mu * z)


def calculate_ND_net_radiation(
    z, ice_thickness, wavelength_in_nm, oil_mass_ratio, ice_type: str
):
    return calculate_ND_downwelling(
        z, ice_thickness, wavelength_in_nm, oil_mass_ratio, ice_type
    ) - calculate_ND_upwelling(
        z, ice_thickness, wavelength_in_nm, oil_mass_ratio, ice_type
    )


def calculate_ND_heating(
    z, ice_thickness, wavelength_in_nm, oil_mass_ratio, ice_type: str
):
    """Use two stream ODEs dFnet/dz = k*(upwelling + downwelling)

    Need to multiply by incident shortwave to get dimensional value
    """
    k = calculate_ice_oil_absorption_coefficient(wavelength_in_nm, oil_mass_ratio)
    upwelling = calculate_ND_upwelling(
        z, ice_thickness, wavelength_in_nm, oil_mass_ratio, ice_type
    )
    downwelling = calculate_ND_downwelling(
        z, ice_thickness, wavelength_in_nm, oil_mass_ratio, ice_type
    )
    return k * (upwelling + downwelling)
