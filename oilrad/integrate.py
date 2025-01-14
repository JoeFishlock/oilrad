"""Spectrally integrate two-stream model solutions
"""

from typing import Optional
import numpy as np
from scipy.integrate import trapezoid

from .constants import (
    CLOUDY_SKY_FRACTIONS,
)
from .spectra import BlackBodySpectrum
from .cts_wavelength import CtsWavelengthIrradiance, CtsWavelengthSpectralIrradiance
from .six_band import SixBandIrradiance, SixBandSpectralIrradiance


def integrate_over_SW(
    spectral_irradiance: CtsWavelengthSpectralIrradiance | SixBandSpectralIrradiance,
    spectrum: Optional[BlackBodySpectrum] = None,
) -> CtsWavelengthIrradiance | SixBandIrradiance:
    """Spectrally integrate the two-stream model solution

    When integrating the continuous wavelength model need to supply an instance of the
    BlackBodySpectrum

    Integration of the six band model uses the cloudy incident spectrum of Grenfell
    et al 2004.

    Args:
        spectral_irradiance (SpectralIrradiance | SixBandSpectralIrradiance):
            spectral two-stream model solution
        spectrum (Optional[BlackBodySpectrum]): normalised incident shortwave spectrum
            (only needed when integrating SpectralIrradiance).
            Currently only the BlackBodySpectrum is implemented.
    Returns:
        Irradiance | SixBandIrradiance: spectrally integrated irradiances
    """
    if isinstance(spectral_irradiance, CtsWavelengthSpectralIrradiance):
        if spectrum is None:
            raise ValueError("spectrum must be provided")
        wavelengths = spectral_irradiance.wavelengths
        integrate = lambda irradiance: trapezoid(
            irradiance * spectrum(wavelengths), wavelengths, axis=1
        )
        integrated_upwelling = integrate(spectral_irradiance.upwelling)
        integrated_downwelling = integrate(spectral_irradiance.downwelling)
        return CtsWavelengthIrradiance(
            spectral_irradiance.z,
            integrated_upwelling,
            integrated_downwelling,
            _ice_base_index=spectral_irradiance._ice_base_index,
        )
    if isinstance(spectral_irradiance, SixBandSpectralIrradiance):
        integrate = lambda x: np.sum(x * CLOUDY_SKY_FRACTIONS, axis=1)
        return SixBandIrradiance(
            spectral_irradiance.z,
            integrate(spectral_irradiance.upwelling),
            integrate(spectral_irradiance.downwelling),
            np.sum(CLOUDY_SKY_FRACTIONS * spectral_irradiance.albedo),
            _ice_base_index=spectral_irradiance._ice_base_index,
        )
    raise NotImplementedError()
