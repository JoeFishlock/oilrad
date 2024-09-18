"""Solve two stream radiation model in layer of ice with continuously varying
vertical profile of mass concentration of oil
"""

from dataclasses import dataclass
from typing import Callable
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_bvp, trapezoid
from .optics import (
    calculate_ice_oil_absorption_coefficient,
    calculate_ice_scattering_coefficient_from_Roche_2022,
)
from .black_body import normalised_black_body_spectrum


@dataclass(frozen=True)
class InfiniteLayerModel:
    z: NDArray
    wavelengths: NDArray
    oil_mass_ratio: Callable[[NDArray], NDArray]
    ice_type: str
    median_droplet_radius_in_microns: float


@dataclass(frozen=True)
class SpectralIrradiance:
    """vertical grid z specified in dimensional units (m)
    discretised wavelengths are in nm and define the SW range
    upwelling and downwelling irradiances are non-dimensional and need to be multiplied
    by the incident radiation spectrum to regain dimensions.
    """

    z: NDArray
    wavelengths: NDArray
    upwelling: NDArray
    downwelling: NDArray

    @property
    def net_irradiance(self) -> NDArray:
        return self.downwelling - self.upwelling

    @property
    def albedo(self) -> NDArray:
        return self.upwelling[-1, :]

    @property
    def transmittance(self) -> NDArray:
        return self.downwelling[0, :]


@dataclass(frozen=True)
class Irradiance:
    """vertical grid z specified in dimensional units (m)
    upwelling and downwelling irradiances are non-dimensional and need to be multiplied
    by the incident integrated SW radiation to regain dimensions.
    """

    z: NDArray
    upwelling: NDArray
    downwelling: NDArray

    @property
    def net_irradiance(self) -> NDArray:
        return self.downwelling - self.upwelling

    @property
    def albedo(self) -> NDArray:
        return self.upwelling[-1]

    @property
    def transmittance(self) -> NDArray:
        return self.downwelling[0]


def integrate_over_SW(spectral_irradiance: SpectralIrradiance) -> Irradiance:
    """integrate over the SW spectrum gven as part of the SpectralIrradiance object
    weighted by the normalised black body spectrum over this range"""
    wavelengths = spectral_irradiance.wavelengths
    integrate = lambda irradiance: trapezoid(
        irradiance * normalised_black_body_spectrum(wavelengths), wavelengths, axis=1
    )
    integrated_upwelling = integrate(spectral_irradiance.upwelling)
    integrated_downwelling = integrate(spectral_irradiance.downwelling)
    return Irradiance(
        spectral_irradiance.z, integrated_upwelling, integrated_downwelling
    )


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


def _solve_at_given_wavelength(model, wavelength: float) -> tuple[NDArray, NDArray]:
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


def solve(model: InfiniteLayerModel) -> SpectralIrradiance:
    upwelling = np.empty((model.z.size, model.wavelengths.size))
    downwelling = np.empty((model.z.size, model.wavelengths.size))
    for i, wavelength in enumerate(model.wavelengths):
        col_upwelling, col_downwelling = _solve_at_given_wavelength(model, wavelength)
        upwelling[:, i] = col_upwelling
        downwelling[:, i] = col_downwelling
    return SpectralIrradiance(model.z, model.wavelengths, upwelling, downwelling)
