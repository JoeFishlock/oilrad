"""Constants used in the program"""
import numpy as np
from numpy.typing import NDArray
from pathlib import Path

PLANCK = 6.62607015e-34  # Js
LIGHTSPEED = 299792458  # m/s
MOLE = 6.023e23
BOLTZMANN = 1.380649e-23  # J/K
AU = 1.496e11  # m
SUN_RADIUS = 6.95e8  # m


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
WAVELENGTH_BAND_INDICES = [0, 1, 2, 3, 4, 5]

# Calculated by digitising figure 11 in Grenfell and Perovich 2004 and integrating
# over the wavelength bands
CLOUDY_SKY_FRACTIONS = np.array([0.086, 0.217, 0.196, 0.155, 0.301, 0.045])


def _read_into_six_bands(
    data_path: Path, interp_right=None, interp_left=None
) -> NDArray:
    """Helper function to read spectral data from csv file and calculate the average
    in each band."""
    data = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    wavelengths = data[:, 0]
    albedo = data[:, 1]
    interpolated_wavelengths = np.linspace(300, 3000, 3000)
    interpolated_albedo = np.interp(
        interpolated_wavelengths,
        wavelengths,
        albedo,
        right=interp_right,
        left=interp_left,
    )

    band_albedo = []
    for band in WAVELENGTH_BANDS:
        left, right = band
        in_band = (interpolated_wavelengths >= left) & (
            interpolated_wavelengths <= right
        )
        band_albedo.append(np.mean(interpolated_albedo[in_band]))
    return np.array(band_albedo)


def _replace_last_value(array: NDArray, value: float) -> NDArray:
    """Helper function to replace the last value of an array with a new value."""
    array[-1] = value
    return array


# Snow and SSL albedos from the literature averaged over
# the six band model spectral bands
DATADIR = Path(__file__).parent / "data/SnowSSL"

_last_band_snow_albedo = 0.1
SNOW_ALBEDOS = {
    "grenfell2004": _read_into_six_bands(
        DATADIR / "Grenfell2004Fig10aColdSnowAlbedo.csv", interp_right=0
    ),
    "grenfell1984": _read_into_six_bands(
        DATADIR / "Grenfell1984Fig3SnowALbedoMay21.csv", interp_right=0
    ),
    # No data beyond 1000 nm
    "verin2022": _replace_last_value(
        _read_into_six_bands(DATADIR / "Verin2022Fig7aColdSnowAlbedo.csv"),
        _last_band_snow_albedo,
    ),
    "light2022": _replace_last_value(
        _read_into_six_bands(DATADIR / "Light2022Fig2ColdSnowAlbedo.csv"),
        _last_band_snow_albedo,
    ),
    "light2022wet": _replace_last_value(
        _read_into_six_bands(DATADIR / "Light2022Fig2MeltingSnowAlbedo.csv"),
        _last_band_snow_albedo,
    ),
}
SSL_ALBEDOS = {
    "smith2022_20": _read_into_six_bands(
        DATADIR / "Smith2022Fig20720.csv", interp_right=0
    ),
    "smith2022_23": _read_into_six_bands(
        DATADIR / "Smith2022Fig20723.csv", interp_right=0
    ),
    "smith2022_24": _read_into_six_bands(
        DATADIR / "Smith2022Fig20724.csv", interp_right=0
    ),
    "smith2022_27": _read_into_six_bands(
        DATADIR / "Smith2022Fig20727.csv", interp_right=0
    ),
    "light2022": _read_into_six_bands(
        DATADIR / "Light2022Fig2BareIceAlbedo.csv", interp_right=0
    ),
}

# Extinction coefficients for six band model surface layer
_large_extinction_value = 1000
SNOW_EXTINCTION_COEFFICIENTS = {
    "perovich1990": _replace_last_value(
        _read_into_six_bands(DATADIR / "PEROVICH1990Fig2DrySnowExtinction.csv"),
        _large_extinction_value,
    ),
    # replace UV and PAR values and use same as peroich1990 above this
    "lebrun2023": np.array([7, 7, 7, 7, 127.7699531, _large_extinction_value]),
    "perovich1990wet": _replace_last_value(
        _read_into_six_bands(DATADIR / "PEROVICH1990Fig2MeltingSnowExtinction.csv"),
        _large_extinction_value,
    ),
    "lebrun2023wet": np.array([5, 5, 5, 5, 88.83449726, _large_extinction_value]),
}

SSL_EXTINCTION_COEFFICIENTS = {
    "perovich1990": _replace_last_value(
        _read_into_six_bands(DATADIR / "PEROVICH1990Fig2SSLExtinction.csv"),
        _large_extinction_value,
    ),
}
