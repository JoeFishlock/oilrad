"""Define an interface here that all solutions of the two stream radiation model
implement. Specifically once a model is initialised from its required parameters it
will provide methods to determine the upwelling radiation, downwelling radiation and
the radiative heating as functions of depth and wavelength. It will also provide
methods for the spectral albedo and transmission."""

import numpy as np
from .single_layer import SingleLayerModel
from .two_layer import TwoLayerModel
from .infinite_layer import InfiniteLayerModel, solve_at_given_wavelength
from .irradiance import SpectralIrradiance


def solve_two_stream_model(
    model: SingleLayerModel | TwoLayerModel | InfiniteLayerModel,
) -> SpectralIrradiance:
    upwelling = np.empty((model.z.size, model.wavelengths.size))
    downwelling = np.empty((model.z.size, model.wavelengths.size))
    match model:
        case SingleLayerModel() | TwoLayerModel():
            for i, wavelength in enumerate(model.wavelengths):
                upwelling[:, i] = model.upwelling(model.z, wavelength)
                downwelling[:, i] = model.downwelling(model.z, wavelength)
        case InfiniteLayerModel(fast_solve=False):
            for i, wavelength in enumerate(model.wavelengths):
                col_upwelling, col_downwelling = solve_at_given_wavelength(
                    model, wavelength
                )
                upwelling[:, i] = col_upwelling
                downwelling[:, i] = col_downwelling
        case InfiniteLayerModel(fast_solve=True):
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
        case _:
            raise NotImplementedError
    return SpectralIrradiance(model.z, model.wavelengths, upwelling, downwelling)
