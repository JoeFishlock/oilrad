"""Constants used in the program"""
import numpy as np

PLANCK = 6.62607015e-34  # Js
LIGHTSPEED = 299792458  # m/s
MOLE = 6.023e23

ICE_DENSITY_ROCHE_2022 = 800  # in kg/m3

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
