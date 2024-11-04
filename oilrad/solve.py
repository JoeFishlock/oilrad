"""Provide a function to solve the two-stream model in the case of continuously varying
optical properties which implements a faster solve approximation for long wavelengths if
the fast_solve parameter of the model is set to True."""

import numpy as np
from .infinite_layer import InfiniteLayerModel, solve_at_given_wavelength
from .irradiance import SpectralIrradiance


def solve_two_stream_model(
    model: InfiniteLayerModel,
) -> SpectralIrradiance:
    """Solve the two-stream model and return an object containing the solution at all
    specified wavelengths

    Args (InfiniteLayerModel):
        model: two-stream model parameters

    Returns:
        SpectralIrradiance: object containing the solution of the two-stream model at each wavelength
    """

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
    return SpectralIrradiance(
        model.z, model.wavelengths, upwelling, downwelling, model._ice_base_index
    )
