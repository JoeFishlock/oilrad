"""Solution of two stream radiative transfer model in single ice layer
containing uniform oil mass concentration

Assume no Fresnel reflection

parameters:
oil_mass_ratio (ng oil/ g ice)
ice_thickness (m)
scattering and absorption coefficients (1/m) from optics module

"""

import numpy as np
from oilrad.optics import (
    calculate_ice_oil_absorption_coefficient,
    calculate_ice_oil_extinction_coefficient,
    calculate_ice_scattering_coefficient_from_Roche_2022,
)
from dataclasses import dataclass
from oilrad.abstract_model import AbstractModel


@dataclass
class SingleLayerModel(AbstractModel):
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

    oil_mass_ratio: float
    ice_thickness: float
    ice_type: str
    median_droplet_radius_in_microns: float

    @property
    def r(self):
        return calculate_ice_scattering_coefficient_from_Roche_2022(self.ice_type)

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
            ice_type=self.ice_type,
            droplet_radius_in_microns=self.median_droplet_radius_in_microns,
        )

    def s(self, L):
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

    def upwelling(self, z, L):
        return self.A(L) * (
            np.exp(self.mu(L) * z)
            - np.exp(-self.mu(L) * z) / np.exp(2 * self.opt_depth(L))
        )

    def downwelling(self, z, L):
        return (self.A(L) / self.s(L)) * np.exp(self.mu(L) * z) + (
            1 - (self.A(L) / self.s(L))
        ) * np.exp(-self.mu(L) * z)

    def net_radiation(self, z, L):
        return self.downwelling(z, L) - self.upwelling(z, L)

    def albedo(self, L):
        """calculate spectral alebdo with no Fresnel reflection
        wavelength in nm
        ice thickness (m)
        oil mass ratio (ng oil/g ice)
        ice_type for scattering coefficient
        """
        return self.r / (self.k(L) + self.r + (self.mu(L) / np.tanh(self.opt_depth(L))))

    def transmittance(self, L):
        return self.downwelling(-self.ice_thickness, L)

    def heating(self, z, L):
        """Use two stream ODEs dFnet/dz = k*(upwelling + downwelling)

        Need to multiply by incident shortwave to get dimensional value
        """
        return self.k(L) * (self.upwelling(z, L) + self.downwelling(z, L))

    def optically_thick_albedo(self, L):
        """calculate spectral alebdo for wavelength in nm in optically thick limit
        oil mass ratio (ng oil/g ice)
        ice_type for scattering coefficient
        """
        return self.s(L)
