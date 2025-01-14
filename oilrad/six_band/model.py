"""Shortwave radiative transfer model for a layer of sea ice with 6 spectral bands.

Optionally the ice may have a melt pond layer, a snow layer and a surface
scattering layer (SSL).
"""

from dataclasses import dataclass
from typing import Callable, Optional, ClassVar, List, Tuple
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_bvp

from ..constants import (
    WAVELENGTH_BANDS,
    ICE_DENSITY_ROCHE_2022,
)
from ..optics import (
    calculate_scattering,
    Romashkino_MAC,
    calculate_ice_absorption_coefficient,
)
from .top_surface import calculate_band_SSL_albedo, calculate_band_surface_transmittance


@dataclass
class SixBandModel:
    """Class containing all the necessary parameters to solve the two-stream shortwave
    radiative transfer model in a domain with continuously varying liquid fraction and
    oil mass ratio with optical properties averaged in the six spectral bands with
    wavelengths:

    300-400nm
    400-500nm
    500-600nm
    600-700nm
    700-1200nm
    1200-3000nm

    Irradiances are scaled by the incident downwelling in each spectral band.

    If no array is provided for liquid fraction, it is assumed to be zero everywhere.
    This corresponds to a completely frozen domain.

    Oil mass ratio is provided in ng oil / g ice and, along with the median droplet radius
    for the oil droplet distribution, is used to calculate the absorption coefficient
    by interpolating data for Romashkino oil from Redmond Roche et al. 2022.

    Args:
        z (NDArray): vertical grid in meters
        oil_mass_ratio (NDArray): array of oil mass ratio in ng oil / g ice on the
            vertical grid
        ice_scattering_coefficient (float): scattering coefficient for ice in 1/m
        median_droplet_radius_in_microns (float): median droplet radius in microns
        absorption_enhancement_factor (float): enhancement factor for oil absorption
            appropriate for the two-stream model
        snow_depth (float): snow depth in meters
        snow_spectral_albedos (NDArray): spectral albedos for the snow layer in each
            band
        snow_extinction_coefficients (NDArray): spectral extinction coefficient for the
            snow layer in each band
        SSL_depth (float): depth of the surface scattering layer in meters
        SSL_spectral_albedos (NDArray): spectral albedos for the SSL in each band
        SSL_extinction_coefficients (NDArray): spectral extinction coefficients for the
            SSL in each band
        liquid_fraction (Optional[NDArray]): liquid fraction array on the vertical grid
    """

    z: NDArray
    oil_mass_ratio: NDArray
    ice_scattering_coefficient: float
    median_droplet_radius_in_microns: float
    absorption_enhancement_factor: float

    snow_depth: float
    snow_spectral_albedos: NDArray
    snow_extinction_coefficients: NDArray

    SSL_depth: float
    SSL_spectral_albedos: NDArray
    SSL_extinction_coefficients: NDArray

    liquid_fraction: Optional[NDArray] = None

    bands: ClassVar[List[Tuple[int, int]]] = WAVELENGTH_BANDS

    def __post_init__(self):
        # initialise liquid fraction as zero everywhere if not provided
        if self.liquid_fraction is None:
            self.liquid_fraction = np.full_like(self.z, 0)

        # find the index of the ice ocean interface
        self._ice_base_index = np.argmax(self.liquid_fraction < 1)

        # Generate band average ice absorption
        UV_adjusted_bands = [(350, 400)] + self.bands[1:]  # Just average UV in 350-400
        self.band_average_ice_absorption = np.array(
            [
                np.mean(
                    calculate_ice_absorption_coefficient(
                        np.linspace(band[0], band[1], 1000)
                    )
                )
                for band in UV_adjusted_bands
            ]
        )
        # Generate band average oil MAC
        self.band_average_Romashkino_MAC = np.array(
            [
                np.mean(
                    Romashkino_MAC(
                        np.linspace(band[0], band[1], 1000),
                        self.median_droplet_radius_in_microns,
                    )
                )
                for band in UV_adjusted_bands
            ]
        )


def _get_ODE_fun(
    model: SixBandModel, wavelength_band_index: int
) -> Callable[[NDArray, NDArray], NDArray]:
    def r(z: NDArray) -> NDArray:
        return calculate_scattering(
            np.interp(z, model.z, model.liquid_fraction, left=np.nan, right=np.nan),
            model.ice_scattering_coefficient,
        )

    def oil_func(z: NDArray) -> NDArray:
        return np.interp(z, model.z, model.oil_mass_ratio, left=np.nan, right=np.nan)

    def k(z: NDArray) -> NDArray:
        mass_ratio_dimensionless = oil_func(z) * 1e-9
        return model.absorption_enhancement_factor * (
            model.band_average_ice_absorption[wavelength_band_index]
            + mass_ratio_dimensionless
            * 1e3
            * ICE_DENSITY_ROCHE_2022
            * model.band_average_Romashkino_MAC[wavelength_band_index]
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
    surface_transmittance = calculate_band_surface_transmittance(
        model, wavelength_band_index
    )

    def _BCs(F_bottom: NDArray, F_top: NDArray) -> NDArray:
        return np.array([F_top[1] - surface_transmittance, F_bottom[0]])

    return _BCs


def solve_a_wavelength_band(
    model: SixBandModel, wavelength_band_index: int
) -> tuple[NDArray, NDArray]:
    """Use the scipy solve_bvp function to solve the two-stream model as a function of
    depth for each wavelength band.

    Args:
        model (SixBandModel): model parameters
        wavelength_band_index (int): index of the wavelength band to solve
    Returns:
        tuple[NDArray, NDArray]: upwelling and downwelling irradiances as functions of depth
    Raises:
        RuntimeError: if the solver does not converge
    """
    # Add radiaition absorbed in SSL into the top of the ice
    if model.snow_depth == 0:
        absorbed_in_SSL = (
            1
            - calculate_band_SSL_albedo(model, wavelength_band_index)
            - calculate_band_surface_transmittance(model, wavelength_band_index)
        )
    else:
        absorbed_in_SSL = 0

    # In high wavelength band just assume all radiation is absorbed at ice surface
    if wavelength_band_index == 5:
        upwelling = np.zeros_like(model.z)
        downwelling = np.zeros_like(model.z)
        downwelling[-1] = (
            calculate_band_surface_transmittance(model, 5) + absorbed_in_SSL
        )
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

    upwelling = solution.sol(model.z)[0]
    downwelling = solution.sol(model.z)[1]
    downwelling[-1] = downwelling[-1] + absorbed_in_SSL
    return upwelling, downwelling
