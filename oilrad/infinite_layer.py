"""Solve two stream radiation model in layer of ice with continuously varying
vertical profile of mass concentration of oil
"""

from dataclasses import dataclass
from typing import Callable
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_bvp
from .optics import (
    calculate_ice_oil_absorption_coefficient,
    calculate_ice_scattering_coefficient_from_Roche_2022,
)


@dataclass(frozen=True)
class InfiniteLayerModel:
    z: NDArray
    wavelengths: NDArray
    oil_mass_ratio: Callable[[NDArray], NDArray]
    ice_type: str
    median_droplet_radius_in_microns: float


def _get_ODE_fun(
    model: InfiniteLayerModel, wavelength: float
) -> Callable[[NDArray, NDArray], NDArray]:
    r = calculate_ice_scattering_coefficient_from_Roche_2022(model.ice_type)

    def k(z: NDArray) -> NDArray:
        return calculate_ice_oil_absorption_coefficient(
            wavelength,
            oil_mass_ratio=model.oil_mass_ratio(z),
            droplet_radius_in_microns=model.median_droplet_radius_in_microns,
        )

    def _ODE_fun(z: NDArray, F: NDArray) -> NDArray:
        """F = [upwelling(z), downwelling(z)]"""
        upwelling_part = -(k(z) + r) * F[0] + r * F[1]
        downwelling_part = (k(z) + r) * F[1] - r * F[0]
        return np.vstack((upwelling_part, downwelling_part))

    return _ODE_fun


def _BCs(F_bottom, F_top):
    """Doesn't depend on wavelength"""
    return np.array([F_top[1] - 1, F_bottom[0]])


def solve_at_given_wavelength(model, wavelength: float) -> tuple[NDArray, NDArray]:
    fun = _get_ODE_fun(model, wavelength)
    solution = solve_bvp(
        fun,
        _BCs,
        np.linspace(model.z[0], model.z[-1], 5),
        np.zeros((2, 5)),
        max_nodes=6000,
    )
    if not solution.success:
        raise RuntimeError(f"{solution.message}")
    return solution.sol(model.z)[0], solution.sol(model.z)[1]
