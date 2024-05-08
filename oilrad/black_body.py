import numpy as np
from scipy.integrate import quad

"""Data from:
https://www.oceanopticsbook.info/view/light-and-radiometry/level-2/light-from-the-sun
"""


EARTH_SOLAR_IRRADIANCE = 1367  # W / m2

NEAR_ULTRAVIOLET = [350, 400]  # nm
VISIBLE = [400, 700]  # nm
NEAR_INFRARED = [700, 1000]  # nm

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


TOTAL_TOP_OF_ATMOSPHERE_IRRADIANCE = quad(top_of_atmosphere_irradiance, 0, np.Inf)[0]


"""Factors to multiply spectrum by for different environmental conditions to get
surface radiation. Note this neglects more complicated atmospheric absorption
for example strong ozone absorption around 300nm.
0: top of atmosphere
1: very clear atmosphere, sun at zenith
2: vlear atmoshphere, sun at 60 deg
3: hazy atmosphere, sun at 60 deg
4: hazy atmosphere, sun near horizon
5: heavy overcast, sun at zenith
6: heavy overcast, sun near horizon

To extend this I could use ERA5 reanalysis for integrated donwelling shortwave

See also observational dataset from greenland of high arctic:
https://doi.org/10.5194/essd-16-543-2024
"""

ENVIRONMENT_FACTORS = [
    1,
    500 / 522,
    250 / 522,
    175 / 522,
    50 / 522,
    125 / 522,
    10 / 522,
]


def solar_irradiance(wavelength, environment_conditions=3):
    return ENVIRONMENT_FACTORS[environment_conditions] * top_of_atmosphere_irradiance(
        wavelength
    )
