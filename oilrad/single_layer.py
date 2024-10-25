"""Solution of two stream radiative transfer model in single ice layer
containing uniform oil mass concentration

Assume no Fresnel reflection

parameters:
oil_mass_ratio (ng oil/ g ice)
ice_thickness (m)
scattering and absorption coefficients (1/m) from optics module

"""

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from .optics import (
    calculate_ice_oil_absorption_coefficient,
    calculate_ice_oil_extinction_coefficient,
)


@dataclass
class SingleLayerModel:
    """
    For computational stability try represnting single layer solution in exponential
    basis:

    upwelling = A exp(+ext z) + B exp(-ext z)
    downwelling = C exp(+ext z) + D exp(-ext z)

    B = -A exp(-2 optical depth)
    C = A/s
    D = 1 - A/s

    s is optically thick single layer albedo
    optical depth = ice_thickness * extinction coefficient

    Non dimensionalise by the incident shortwave at each wavelength so that
    upwelling radiation -> upwelling * incident shortwave spectrum
    """

    z: NDArray
    wavelengths: NDArray

    oil_mass_ratio: float
    ice_scattering_coefficient: float  # in 1/m
    median_droplet_radius_in_microns: float

    @property
    def ice_thickness(self):
        return -self.z[0]

    @property
    def r(self):
        return self.ice_scattering_coefficient

    def k(self, L):
        return calculate_ice_oil_absorption_coefficient(
            L,
            oil_mass_ratio=self.oil_mass_ratio,
            droplet_radius_in_microns=self.median_droplet_radius_in_microns,
        )

    def mu(self, L):
        return calculate_ice_oil_extinction_coefficient(
            L,
            oil_mass_ratio=self.oil_mass_ratio,
            ice_scattering_coefficient=self.ice_scattering_coefficient,
            droplet_radius_in_microns=self.median_droplet_radius_in_microns,
        )

    def s(self, L):
        """albedo in optically thick limit"""
        return (self.mu(L) - self.k(L)) / (self.mu(L) + self.k(L))

    def opt_depth(self, L):
        return self.mu(L) * self.ice_thickness

    def A(self, L):
        return self.s(L) * (
            1
            / (
                1
                - np.exp(-2 * self.opt_depth(L))
                * (
                    (self.k(L) + self.r - self.mu(L))
                    / (self.k(L) + self.r + self.mu(L))
                )
            )
        )

    def albedo(self, L):
        """calculate spectral alebdo with no Fresnel reflection
        wavelength in nm
        ice thickness (m)
        oil mass ratio (ng oil/g ice)
        ice_type for scattering coefficient
        """
        return self.r / (self.k(L) + self.r + (self.mu(L) / np.tanh(self.opt_depth(L))))

    def optically_thick_albedo(self, L):
        """calculate spectral alebdo for wavelength in nm in optically thick limit
        oil mass ratio (ng oil/g ice)
        ice_type for scattering coefficient
        """
        return self.s(L)

    def upwelling(self, z, L):
        return self.A(L) * (
            np.exp(self.mu(L) * z) - np.exp(-self.mu(L) * (z + 2 * self.ice_thickness))
        )

    def downwelling(self, z, L):
        down = (self.A(L) / self.s(L)) * np.exp(self.mu(L) * z) + (
            1 - (self.A(L) / self.s(L))
        ) * np.exp(-self.mu(L) * z)
        # When downwelling becomes very small as extinction coefficient goes to infinity
        # at long wavelengths
        down[np.isnan(down)] = 0
        return down
