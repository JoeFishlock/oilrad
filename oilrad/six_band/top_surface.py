"""Calculate surface transmittance for six band model through a snow layer and SSL"""
import numpy as np


def calculate_band_surface_albedo(model, wavelength_band_index: int) -> float:
    _decay_length = 0.02
    return model.snow_spectral_albedos[wavelength_band_index] * (
        1 - np.exp(-model.snow_depth / _decay_length)
    ) + calculate_band_SSL_albedo(model, wavelength_band_index) * np.exp(
        -model.snow_depth / _decay_length
    )


def calculate_band_surface_transmittance(model, wavelength_band_index: int) -> float:
    albedo = calculate_band_surface_albedo(
        model,
        wavelength_band_index,
    )
    attenuation = min(
        np.exp(
            -model.snow_extinction_coefficients[wavelength_band_index]
            * model.snow_depth
        ),
        np.exp(
            -model.SSL_extinction_coefficients[wavelength_band_index] * model.SSL_depth
        ),
    )
    return (1 - albedo) * attenuation


def calculate_band_SSL_albedo(model, wavelength_band_index: int) -> float:
    if model.SSL_depth > 0:
        return model.SSL_spectral_albedos[wavelength_band_index]
    elif model.SSL_depth == 0:
        return 0
    else:
        raise ValueError("SSL depth must be non-negative")
