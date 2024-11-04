"""Define an interface here that all solutions of the two stream radiation model
implement. Specifically once a model is initialised from its required parameters it
will provide methods to determine the upwelling radiation, downwelling radiation and
the radiative heating as functions of depth and wavelength. It will also provide
methods for the spectral albedo and transmission."""

import numpy as np
from .infinite_layer import InfiniteLayerModel, solve_at_given_wavelength
from .irradiance import SpectralIrradiance


def solve_two_stream_model(
    model: InfiniteLayerModel,
) -> SpectralIrradiance:
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
        model.z, model.wavelengths, upwelling, downwelling, model.ice_base_index
    )
