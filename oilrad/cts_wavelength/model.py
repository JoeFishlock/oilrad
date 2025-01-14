"""Solve the two-stream shortwave radiation model for a layer of ice with
constant optical properties aside from absorption varying due to the oil mass ratio.

Arbitrary wavelength resolution
"""

from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_bvp
from ..optics import (
    calculate_ice_oil_absorption_coefficient,
    calculate_scattering,
)


@dataclass
class CtsWavelengthModel:
    """Class containing all the necessary parameters to solve the two-stream shortwave
    radiative transfer model in a domain with continuously varying liquid fraction and
    oil mass ratio.

    If no array is provided for liquid fraction, it is assumed to be zero everywhere.
    This corresponds to a completely frozen domain.

    Oil mass ratio is provided in ng oil / g ice and, along with the median droplet radius
    for the oil droplet distribution, is used to calculate the absorption coefficient
    by interpolating data for Romashkino oil from Redmond Roche et al. 2022.

    Args:
        z (NDArray): vertical grid in meters
        wavelengths (NDArray): array of wavelengths in nm
        oil_mass_ratio (NDArray): array of oil mass ratio in ng oil / g ice on the vertical grid
        ice_scattering_coefficient (float): scattering coefficient for ice in 1/m
        median_droplet_radius_in_microns (float): median droplet radius in microns
        absorption_enhancement_factor (float): enhancement factor for oil absorption appropriate for the two-stream model
        liquid_fraction (NDArray): liquid fraction array on the vertical grid
        fast_solve (Bool): if True, solve the model only for wavelengths below a wavelength cutoff, assume longer wavelengths are absorbed at the surface
        wavelength_cutoff (float): cutoff wavelength in nm
    """

    z: NDArray
    wavelengths: NDArray
    oil_mass_ratio: NDArray
    ice_scattering_coefficient: float
    median_droplet_radius_in_microns: float
    absorption_enhancement_factor: float = 1
    liquid_fraction: Optional[NDArray] = None
    fast_solve: bool = False
    wavelength_cutoff: Optional[float] = None

    def __post_init__(self):
        # initialise liquid fraction as zero everywhere if not provided
        if self.liquid_fraction is None:
            self.liquid_fraction = np.full_like(self.z, 0)

        # find the index of the ice ocean interface
        self._ice_base_index = np.argmax(self.liquid_fraction < 1)


def _get_ODE_fun(
    model: CtsWavelengthModel, wavelength: float
) -> Callable[[NDArray, NDArray], NDArray]:
    def r(z: NDArray) -> NDArray:
        return calculate_scattering(
            np.interp(z, model.z, model.liquid_fraction, left=np.nan, right=np.nan),
            model.ice_scattering_coefficient,
        )

    def oil_func(z: NDArray) -> NDArray:
        return np.interp(z, model.z, model.oil_mass_ratio, left=np.nan, right=np.nan)

    def k(z: NDArray) -> NDArray:
        return calculate_ice_oil_absorption_coefficient(
            wavelength,
            oil_mass_ratio=oil_func(z),
            droplet_radius_in_microns=model.median_droplet_radius_in_microns,
            absorption_enhancement_factor=model.absorption_enhancement_factor,
        )

    def _ODE_fun(z: NDArray, F: NDArray) -> NDArray:
        # F = [upwelling(z), downwelling(z)]
        upwelling_part = -(k(z) + r(z)) * F[0] + r(z) * F[1]
        downwelling_part = (k(z) + r(z)) * F[1] - r(z) * F[0]
        return np.vstack((upwelling_part, downwelling_part))

    return _ODE_fun


def _BCs(F_bottom, F_top):
    # Doesn't depend on wavelength
    return np.array([F_top[1] - 1, F_bottom[0]])


def solve_at_given_wavelength(model, wavelength: float) -> tuple[NDArray, NDArray]:
    """Use the scipy solve_bvp function to solve the two-stream model as a function of
    depth for a given wavelenght value.

    Args:
        model (CtsWavelengthModel): model parameters
        wavelength (float): wavelength in nm
    Returns:
        tuple[NDArray, NDArray]: upwelling and downwelling irradiances as functions of depth
    Raises:
        RuntimeError: if the solver does not converge
    """
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
