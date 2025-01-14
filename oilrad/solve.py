"""Provide a function to solve the two-stream model in the case of continuously varying
optical properties which implements a faster solve approximation for long wavelengths if
the fast_solve parameter of the model is set to True for the continuous wavelength
case."""

import numpy as np

from .cts_wavelength import (
    CtsWavelengthModel,
    solve_at_given_wavelength,
    CtsWavelengthSpectralIrradiance,
)
from .constants import WAVELENGTH_BAND_INDICES
from .six_band import SixBandModel, solve_a_wavelength_band, SixBandSpectralIrradiance
from .six_band.top_surface import (
    calculate_band_surface_albedo,
    calculate_band_surface_transmittance,
)


def solve_two_stream_model(
    model: CtsWavelengthModel | SixBandModel,
) -> CtsWavelengthSpectralIrradiance | SixBandSpectralIrradiance:
    """Solve the two-stream model and return an object containing the solution at all
    specified wavelengths

    Args (InfiniteLayerModel | SixBandModel):
        model: two-stream model parameters

    Returns:
        SpectralIrradiance | SixBandSpectralIrradiance: object containing the solution of the two-stream model at each wavelength
    """

    if isinstance(model, CtsWavelengthModel):
        upwelling = np.empty((model.z.size, model.wavelengths.size))
        downwelling = np.empty((model.z.size, model.wavelengths.size))
        if model.fast_solve:
            cut_off_index = (
                np.argmin(np.abs(model.wavelengths - model.wavelength_cutoff)) + 1
            )
            is_surface = np.s_[cut_off_index:]
            is_interior = np.s_[:cut_off_index]
            for i, wavelength in enumerate(model.wavelengths[is_interior]):
                col_upwelling, col_downwelling = solve_at_given_wavelength(
                    model, wavelength
                )
                upwelling[:, i] = col_upwelling
                downwelling[:, i] = col_downwelling

            upwelling[:, is_surface] = 0
            downwelling[:, is_surface] = 0
            downwelling[-1, is_surface] = 1
        else:
            for i, wavelength in enumerate(model.wavelengths):
                col_upwelling, col_downwelling = solve_at_given_wavelength(
                    model, wavelength
                )
                upwelling[:, i] = col_upwelling
                downwelling[:, i] = col_downwelling
        return CtsWavelengthSpectralIrradiance(
            model.z, model.wavelengths, upwelling, downwelling, model._ice_base_index
        )

    if isinstance(model, SixBandModel):
        upwelling = np.empty((model.z.size, 6))
        downwelling = np.empty((model.z.size, 6))
        for index in WAVELENGTH_BAND_INDICES:
            col_upwelling, col_downwelling = solve_a_wavelength_band(model, index)
            upwelling[:, index] = col_upwelling
            downwelling[:, index] = col_downwelling

        ice_albedo = upwelling[-1, :]
        surface_albedo = np.array(
            [calculate_band_surface_albedo(model, i) for i in WAVELENGTH_BAND_INDICES]
        )
        surface_transmittance = np.array(
            [
                calculate_band_surface_transmittance(model, i)
                for i in WAVELENGTH_BAND_INDICES
            ]
        )
        albedo = surface_albedo + surface_transmittance * ice_albedo
        return SixBandSpectralIrradiance(
            model.z,
            upwelling,
            downwelling,
            albedo,
            model._ice_base_index,
        )

    raise NotImplementedError("Model type not recognized")
