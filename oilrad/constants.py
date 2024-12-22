"""Constants used in the program"""
import numpy as np

PLANCK = 6.62607015e-34  # Js
LIGHTSPEED = 299792458  # m/s
MOLE = 6.023e23

# wavelength bands for six band model
WAVELENGTH_BANDS = [
    (300, 400),
    (400, 500),
    (500, 600),
    (600, 700),
    (700, 1200),
    (1200, 3000),
]
WAVELENGTH_WIDTHS = np.array([100, 100, 100, 100, 500, 1800])
WAVELENGTH_BAND_INDICES = [0, 1, 2, 3, 4, 5]

# Calculated by digitising figure 11 in Grenfell and Perovich 2004 and integrating
# over the wavelength bands
CLOUDY_SKY_FRACTIONS = np.array([0.086, 0.217, 0.196, 0.155, 0.301, 0.045])


# Optical properties for snow and SSL for wavelength bands in six band model

# In visible bands take from fig2 Perovich 1990 (SSL and dry snow):
# UV band extrapolate value at 400nm
# PAR bands use the value at the midpoint (i.e. 450nm 550nm 650nm)
# For 700-1200nm band use vvalue at 950nm
# For 1200-3000nm band just set value to infinity
_large_value = 1000
SSL_EXTINCTION_COEFFICIENTS = np.array([3.02, 3.09, 4.21, 7.70, 85.56, _large_value])
SNOW_EXTINCTION_COEFFICIENTS = np.array(
    [16.02, 15.55, 18.7, 30.15, 125.18, _large_value]
)

# Optically thick albedos

# From Fig10 in Grenfell and Perovich 2004 digitised values at 350nm 450nm 550nm 650nm 950nm
# The value for the last band is set to 0
SNOW_SPECTRAL_ALBEDOS = np.array([0.92, 0.96, 0.98, 0.98, 0.89, 0])

# From Fig2 in Smith2022 from the ice before shovelling (4cm thick SSL present)
# digitised values at 350nm, 450nm, 550nm, 650nm, 950nm and high band set to zero
SSL_SPECTRAL_ALBEDOS = np.array([0.79, 0.8, 0.79, 0.75, 0.5, 0])


def calculate_band_snow_albedo(snow_depth: float, wavelength_band_index: int) -> float:
    _decay_length = 0.02
    return SNOW_SPECTRAL_ALBEDOS[wavelength_band_index] * (
        1 - np.exp(-snow_depth / _decay_length)
    )


def calculate_band_snow_transmittance(
    snow_depth: float, wavelength_band_index: int
) -> float:
    return (1 - calculate_band_snow_albedo(snow_depth, wavelength_band_index)) * np.exp(
        -SNOW_EXTINCTION_COEFFICIENTS[wavelength_band_index] * snow_depth
    )


def calculate_band_SSL_albedo(SSL_depth: float, wavelength_band_index: int) -> float:
    if SSL_depth > 0:
        return SSL_SPECTRAL_ALBEDOS[wavelength_band_index]
    elif SSL_depth == 0:
        return 0
    else:
        raise ValueError("SSL depth must be non-negative")


def calculate_band_SSL_transmittance(
    SSL_depth: float, wavelength_band_index: int
) -> float:
    return (1 - calculate_band_SSL_albedo(SSL_depth, wavelength_band_index)) * np.exp(
        -SSL_EXTINCTION_COEFFICIENTS[wavelength_band_index] * SSL_depth
    )
