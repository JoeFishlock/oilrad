"""Two-stream shortwave radiative transfer model for sea ice containing oil droplets."""
__version__ = "1.0.0"

from .constants import (
    WAVELENGTH_BANDS,
    SNOW_ALBEDOS,
    SNOW_EXTINCTION_COEFFICIENTS,
    SSL_EXTINCTION_COEFFICIENTS,
    SSL_ALBEDOS,
)
from .irradiance import (
    SpectralIrradiance,
    Irradiance,
    integrate_over_SW,
    SixBandIrradiance,
    SixBandSpectralIrradiance,
)
from .spectra import BlackBodySpectrum
from .infinite_layer import InfiniteLayerModel
from .six_band import SixBandModel
from .solve import solve_two_stream_model
