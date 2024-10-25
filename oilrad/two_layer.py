"""Solution of two stream radiative transfer model in two layer ice
The top layer occupies a fraction f of the domain and contains all the oil
so that if the uniform model oil concentration was X then the top layer in this model
contans X/f mass concentration of oil.

The bottom layer contains no oil.

Assume no Fresnel reflection

parameters:
oil_mass_ratio (ng oil/ g ice)
ice_thickness (m)
scattering and absorption coefficients (1/m) from optics module
oil layer depth fraction f (dimensionless)

gamma is defined as the single layer albedo i.e
gamma = scattering /(extinction*coth(optical_depth) + absorption + scattering)
"""

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from .optics import (
    calculate_ice_oil_absorption_coefficient,
    calculate_ice_oil_extinction_coefficient,
)


@dataclass
class TwoLayerModel:
    z: NDArray
    wavelengths: NDArray

    oil_mass_ratio: float
    thickness_ratio: float
    ice_scattering_coefficient: float  # in 1/m
    median_droplet_radius_in_microns: float

    @property
    def ice_thickness(self):
        return -self.z[0]

    @property
    def top_oil_mass_ratio(self):
        return self.oil_mass_ratio / self.thickness_ratio

    @property
    def r(self):
        return self.ice_scattering_coefficient

    def k1(self, L):
        return calculate_ice_oil_absorption_coefficient(
            L,
            oil_mass_ratio=self.top_oil_mass_ratio,
            droplet_radius_in_microns=self.median_droplet_radius_in_microns,
        )

    def k2(self, L):
        return calculate_ice_oil_absorption_coefficient(
            L,
            oil_mass_ratio=0,
            droplet_radius_in_microns=self.median_droplet_radius_in_microns,
        )

    def mu1(self, L):
        return calculate_ice_oil_extinction_coefficient(
            L,
            oil_mass_ratio=self.top_oil_mass_ratio,
            ice_scattering_coefficient=self.ice_scattering_coefficient,
            droplet_radius_in_microns=self.median_droplet_radius_in_microns,
        )

    def mu2(self, L):
        return calculate_ice_oil_extinction_coefficient(
            L,
            oil_mass_ratio=0,
            ice_scattering_coefficient=self.ice_scattering_coefficient,
            droplet_radius_in_microns=self.median_droplet_radius_in_microns,
        )

    def s1(self, L):
        return (self.mu1(L) - self.k1(L)) / (self.mu1(L) + self.k1(L))

    def s2(self, L):
        return (self.mu2(L) - self.k2(L)) / (self.mu2(L) + self.k2(L))

    def optical_depth1(self, L):
        return self.thickness_ratio * self.ice_thickness * self.mu1(L)

    def optical_depth2(self, L):
        return (1 - self.thickness_ratio) * self.ice_thickness * self.mu2(L)

    def gamma1(self, L):
        return self.r / (
            (self.mu1(L) / np.tanh(self.optical_depth1(L))) + self.k1(L) + self.r
        )

    def gamma2(self, L):
        return self.r / (
            (self.mu2(L) / np.tanh(self.optical_depth2(L))) + self.k2(L) + self.r
        )

    def A2(self, L):
        return (
            (self.r / self.mu1(L))
            * np.sinh(self.optical_depth1(L))
            * ((self.albedo(L) / self.gamma1(L)) - 1)
        )

    def albedo(self, L):
        numerator = (self.mu1(L) / np.tanh(self.optical_depth1(L))) - (
            self.k1(L) + self.r
        )
        denominator = (self.mu2(L) / np.tanh(self.optical_depth2(L))) + (
            self.k2(L) + self.r
        )
        return (self.gamma1(L) / (1 - self.gamma1(L) * self.gamma2(L))) * (
            1 + (numerator / denominator)
        )

    def _upwelling_1(self, z, L):
        return (self.r / (2 * self.mu1(L))) * (
            (1 - self.albedo(L) * self.s1(L)) * np.exp(self.mu1(L) * z)
            + ((self.albedo(L) / self.s1(L)) - 1) * np.exp(-self.mu1(L) * z)
        )

    def _downwelling_1(self, z, L):
        return (self.r / (2 * self.mu1(L))) * (
            ((1 / self.s1(L)) - self.albedo(L)) * np.exp(self.mu1(L) * z)
            + (self.albedo(L) - self.s1(L)) * np.exp(-self.mu1(L) * z)
        )

    def _upwelling_2(self, z, L):
        return (self.A2(L) / 2) * (
            (1 + (1 / np.tanh(self.optical_depth2(L)))) * np.exp(self.mu2(L) * z)
            + (1 - (1 / np.tanh(self.optical_depth2(L)))) * np.exp(-self.mu2(L) * z)
        )

    def _downwelling_2(self, z, L):
        return (self.A2(L) / 2) * (
            ((1 + (1 / np.tanh(self.optical_depth2(L)))) / self.s2(L))
            * np.exp(self.mu2(L) * z)
            - self.s2(L)
            * ((1 / np.tanh(self.optical_depth2(L))) - 1)
            * np.exp(-self.mu2(L) * z)
        )

    def downwelling(self, z, L):
        return _make_piecewise(
            self._downwelling_1,
            self._downwelling_2,
            boundary=-self.ice_thickness * self.thickness_ratio,
        )(z, L)

    def upwelling(self, z, L):
        return _make_piecewise(
            self._upwelling_1,
            self._upwelling_2,
            boundary=-self.ice_thickness * self.thickness_ratio,
        )(z, L)

    @property
    def k_cts(self):
        def piecewise(z, L):
            output = np.empty_like(z)
            is_region_1 = z >= -self.thickness_ratio * self.ice_thickness
            output[is_region_1] = self.k1(L)
            output[~is_region_1] = self.k2(L)
            return output

        return piecewise

    def heating(self, z, L):
        return self.k_cts(z, L) * (self.upwelling(z, L) + self.downwelling(z, L))

    def transmittance(self, L):
        return self.downwelling(-self.ice_thickness, L)


def _make_piecewise(func1, func2, boundary):
    def piecewise_function(z, L):
        output = np.empty_like(z)
        is_region_1 = z >= boundary
        output[is_region_1] = func1(z[is_region_1], L)
        output[~is_region_1] = func2(z[~is_region_1] - boundary, L)
        return output

    return piecewise_function
