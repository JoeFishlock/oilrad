"""Shortwave radiative transfer model for a layer of sea ice with 6 spectral bands.

Optionally the ice may have a melt pond layer, a snow layer and a surface
scattering layer (SSL).
"""

from dataclasses import dataclass
from typing import Callable, Optional, ClassVar, List, Tuple
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_bvp

from .constants import (
    WAVELENGTH_BANDS,
    calculate_band_snow_transmittance,
    calculate_band_SSL_transmittance,
)
from .optics import (
    calculate_ice_oil_absorption_coefficient,
    calculate_scattering,
)


@dataclass
class SixBandModel:
    """Class containing all the necessary parameters to solve the two-stream shortwave
    radiative transfer model in a domain with continuously varying liquid fraction and
    oil mass ratio.

    Irradiances are scaled by the incident downwelling in each spectral band with
    treatment of any snow layer and surface scattering layer (SSL) handled by the model.

    If no array is provided for liquid fraction, it is assumed to be zero everywhere.
    This corresponds to a completely frozen domain.

    Oil mass ratio is provided in ng oil / g ice and, along with the median droplet radius
    for the oil droplet distribution, is used to calculate the absorption coefficient
    by interpolating data for Romashkino oil from Redmond Roche et al. 2022.

    Args:
        z (NDArray): vertical grid in meters
        oil_mass_ratio (NDArray): array of oil mass ratio in ng oil / g ice on the vertical grid
        ice_scattering_coefficient (float): scattering coefficient for ice in 1/m
        median_droplet_radius_in_microns (float): median droplet radius in microns
        absorption_enhancement_factor (float): enhancement factor for oil absorption appropriate for the two-stream model
        SSL_depth (float): depth of the surface scattering layer in meters
        snow_depth (float): snow depth in meters
        liquid_fraction (Optional[NDArray]): liquid fraction array on the vertical grid
    """

    z: NDArray
    oil_mass_ratio: NDArray
    ice_scattering_coefficient: float
    median_droplet_radius_in_microns: float
    absorption_enhancement_factor: float
    SSL_depth: float
    snow_depth: float
    liquid_fraction: Optional[NDArray] = None

    bands: ClassVar[List[Tuple[int, int]]] = WAVELENGTH_BANDS

    def __post_init__(self):
        # initialise liquid fraction as zero everywhere if not provided
        if self.liquid_fraction is None:
            self.liquid_fraction = np.full_like(self.z, 0)

        # find the index of the ice ocean interface
        self._ice_base_index = np.argmax(self.liquid_fraction < 1)


def _get_ODE_fun(
    model: SixBandModel, wavelength_band_index: int
) -> Callable[[NDArray, NDArray], NDArray]:

    # wavelengths at which to evaluate oil and ice absorption
    wavelengths = np.array([350, 450, 550, 650, 950])

    def r(z: NDArray) -> NDArray:
        return calculate_scattering(
            np.interp(z, model.z, model.liquid_fraction, left=np.nan, right=np.nan),
            model.ice_scattering_coefficient,
        )

    def oil_func(z: NDArray) -> NDArray:
        return np.interp(z, model.z, model.oil_mass_ratio, left=np.nan, right=np.nan)

    def k(z: NDArray) -> NDArray:
        return calculate_ice_oil_absorption_coefficient(
            wavelengths[wavelength_band_index],
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


def _get_BC_fun(
    model: SixBandModel, wavelength_band_index: int
) -> Callable[[NDArray, NDArray], NDArray]:
    surface_transmittance = calculate_band_snow_transmittance(
        model.snow_depth, wavelength_band_index
    ) * calculate_band_SSL_transmittance(model.SSL_depth, wavelength_band_index)

    def _BCs(F_bottom: NDArray, F_top: NDArray) -> NDArray:
        return np.array([F_top[1] - surface_transmittance, F_bottom[0]])

    return _BCs


def solve_a_wavelength_band(
    model: SixBandModel, wavelength_band_index: int
) -> tuple[NDArray, NDArray]:
    """Use the scipy solve_bcp function to solve the two-stream model as a function of
    depth for each wavelength band.

    Args:
        model (SixBandModel): model parameters
        wavelength_band_index (int): index of the wavelength band to solve
    Returns:
        tuple[NDArray, NDArray]: upwelling and downwelling irradiances as functions of depth
    Raises:
        RuntimeError: if the solver does not converge
    """
    # In high wavelength band just assume all radiation is absorbed at ice surface
    # (including in SSL)
    if wavelength_band_index == 5:
        upwelling = np.zeros_like(model.z)
        downwelling = np.zeros_like(model.z)
        downwelling[-1] = calculate_band_snow_transmittance(model.snow_depth, 5)
        return upwelling, downwelling

    fun = _get_ODE_fun(model, wavelength_band_index)
    BCs = _get_BC_fun(model, wavelength_band_index)
    solution = solve_bvp(
        fun,
        BCs,
        np.linspace(model.z[0], model.z[-1], 5),
        np.zeros((2, 5)),
        max_nodes=12000,
    )
    if not solution.success:
        raise RuntimeError(f"{solution.message}")
    return solution.sol(model.z)[0], solution.sol(model.z)[1]
