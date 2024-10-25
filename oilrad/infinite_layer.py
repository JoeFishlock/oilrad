"""Solve two stream radiation model in layer of ice with continuously varying
vertical profile of mass concentration of oil
"""

from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_bvp
from .optics import (
    calculate_ice_oil_absorption_coefficient,
    calculate_scattering,
)


@dataclass
class InfiniteLayerModel:
    z: NDArray
    wavelengths: NDArray
    oil_mass_ratio: NDArray

    ice_scattering_coefficient: float  # in 1/m
    median_droplet_radius_in_microns: float

    absorption_enhancement_factor: float = 1
    liquid_fraction: Optional[NDArray] = None

    fast_solve: bool = False
    wavelength_cutoff: Optional[float] = None

    def __post_init__(self):
        """if liquid fraction not passed initialise so domain is entirely ice"""
        if self.liquid_fraction is None:
            self.liquid_fraction = np.full_like(self.z, 0)

        self.ice_base_index = np.argmax(self.liquid_fraction < 1)


def _get_ODE_fun(
    model: InfiniteLayerModel, wavelength: float
) -> Callable[[NDArray, NDArray], NDArray]:
    def r(z: NDArray) -> NDArray:
        return calculate_scattering(
            np.interp(z, model.z, model.liquid_fraction, left=np.NaN, right=np.NaN),
            model.ice_scattering_coefficient,
        )

    def oil_func(z: NDArray) -> NDArray:
        return np.interp(z, model.z, model.oil_mass_ratio, left=np.NaN, right=np.NaN)

    def k(z: NDArray) -> NDArray:
        return calculate_ice_oil_absorption_coefficient(
            wavelength,
            oil_mass_ratio=oil_func(z),
            droplet_radius_in_microns=model.median_droplet_radius_in_microns,
            absorption_enhancement_factor=model.absorption_enhancement_factor,
        )

    def _ODE_fun(z: NDArray, F: NDArray) -> NDArray:
        """F = [upwelling(z), downwelling(z)]"""
        upwelling_part = -(k(z) + r(z)) * F[0] + r(z) * F[1]
        downwelling_part = (k(z) + r(z)) * F[1] - r(z) * F[0]
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
        max_nodes=12000,
    )
    if not solution.success:
        raise RuntimeError(f"{solution.message}")
    return solution.sol(model.z)[0], solution.sol(model.z)[1]
