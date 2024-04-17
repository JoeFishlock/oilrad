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

import numpy as np
from gulf.optics import (
    calculate_ice_oil_absorption_coefficient,
    calculate_ice_oil_extinction_coefficient,
    calculate_ice_scattering_coefficient_from_Roche_2022,
)
from dataclasses import dataclass

########################
#  Albedo calculation  #
########################


def calculate_gamma(r, k, mu, layer_thickness):
    return r / ((mu / np.tanh(mu * layer_thickness)) + k + r)


def calculate_albedo(
    ice_thickness, wavelength_in_nm, oil_mass_ratio, ice_type: str, thickness_ratio
):
    """calculate spectral alebdo for wavelength in nm in two layer ice.
    oil mass ratio (ng oil/g ice) in uniform model
    ice_type for scattering coefficient
    ice thickness ratio for oil layer
    """
    r = calculate_ice_scattering_coefficient_from_Roche_2022(ice_type)

    top_oil = oil_mass_ratio / thickness_ratio
    k1 = calculate_ice_oil_absorption_coefficient(wavelength_in_nm, top_oil)
    mu1 = calculate_ice_oil_extinction_coefficient(wavelength_in_nm, top_oil, ice_type)

    k2 = calculate_ice_oil_absorption_coefficient(wavelength_in_nm, 0)
    mu2 = calculate_ice_oil_extinction_coefficient(wavelength_in_nm, 0, ice_type)

    gamma1 = calculate_gamma(r, k1, mu1, ice_thickness * thickness_ratio)
    gamma2 = calculate_gamma(r, k2, mu2, ice_thickness * (1 - thickness_ratio))

    optical_depth1 = ice_thickness * thickness_ratio * mu1
    optical_depth2 = ice_thickness * (1 - thickness_ratio) * mu2

    numerator = (mu1 / np.tanh(optical_depth1)) - (k1 + r)
    denominator = (mu2 / np.tanh(optical_depth2)) + (k2 + r)

    return (gamma1 / (1 - gamma1 * gamma2)) * (1 + (numerator / denominator))


#######################################
#  Upwelling / downwelling radiation  #
#######################################


@dataclass
class TwoLayerModel:
    uniform_oil_mass_ratio: float
    thickness_ratio: float
    ice_thickness: float
    ice_type: str

    @property
    def top_oil_mass_ratio(self):
        return self.uniform_oil_mass_ratio / self.thickness_ratio

    @property
    def r(self):
        return calculate_ice_scattering_coefficient_from_Roche_2022(self.ice_type)

    @property
    def k1(self):
        return lambda L: calculate_ice_oil_absorption_coefficient(
            L, oil_mass_ratio=self.top_oil_mass_ratio
        )

    @property
    def k2(self):
        return lambda L: calculate_ice_oil_absorption_coefficient(L, oil_mass_ratio=0)

    @property
    def mu1(self):
        return lambda L: calculate_ice_oil_extinction_coefficient(
            L, oil_mass_ratio=self.top_oil_mass_ratio, ice_type=self.ice_type
        )

    @property
    def mu2(self):
        return lambda L: calculate_ice_oil_extinction_coefficient(
            L, oil_mass_ratio=0, ice_type=self.ice_type
        )

    @property
    def s1(self):
        return lambda L: (self.mu1(L) - self.k1(L)) / (self.mu1(L) + self.k1(L))

    @property
    def s2(self):
        return lambda L: (self.mu2(L) - self.k2(L)) / (self.mu2(L) + self.k2(L))

    @property
    def optical_depth1(self):
        return lambda L: self.thickness_ratio * self.ice_thickness * self.mu1(L)

    @property
    def optical_depth2(self):
        return lambda L: (1 - self.thickness_ratio) * self.ice_thickness * self.mu2(L)

    @property
    def gamma1(self):
        return lambda L: self.r / (
            (self.mu1(L) / np.tanh(self.optical_depth1(L))) + self.k1(L) + self.r
        )

    @property
    def gamma2(self):
        return lambda L: self.r / (
            (self.mu2(L) / np.tanh(self.optical_depth2(L))) + self.k2(L) + self.r
        )

    @property
    def A2(self):
        return (
            lambda L: (self.r / self.mu1(L))
            * np.sinh(self.optical_depth1(L))
            * ((self.albedo(L) / self.gamma1(L)) - 1)
        )

    @property
    def albedo(self):
        numerator = lambda L: (self.mu1(L) / np.tanh(self.optical_depth1(L))) - (
            self.k1(L) + self.r
        )
        denominator = lambda L: (self.mu2(L) / np.tanh(self.optical_depth2(L))) + (
            self.k2(L) + self.r
        )
        return lambda L: (self.gamma1(L) / (1 - self.gamma1(L) * self.gamma2(L))) * (
            1 + (numerator(L) / denominator(L))
        )

    @property
    def _upwelling_1(self):
        return lambda z, L: (self.r / (2 * self.mu1(L))) * (
            (1 - self.albedo(L) * self.s1(L)) * np.exp(self.mu1(L) * z)
            + ((self.albedo(L) / self.s1(L)) - 1) * np.exp(-self.mu1(L) * z)
        )

    @property
    def _downwelling_1(self):
        return lambda z, L: (self.r / (2 * self.mu1(L))) * (
            ((1 / self.s1(L)) - self.albedo(L)) * np.exp(self.mu1(L) * z)
            + (self.albedo(L) - self.s1(L)) * np.exp(-self.mu1(L) * z)
        )

    @property
    def _upwelling_2(self):
        return lambda z, L: (self.A2(L) / 2) * (
            (1 + (1 / np.tanh(self.optical_depth2(L)))) * np.exp(self.mu2(L) * z)
            + (1 - (1 / np.tanh(self.optical_depth2(L)))) * np.exp(-self.mu2(L) * z)
        )

    @property
    def _downwelling_2(self):
        return lambda z, L: (self.A2(L) / 2) * (
            ((1 + (1 / np.tanh(self.optical_depth2(L)))) / self.s2(L))
            * np.exp(self.mu2(L) * z)
            - self.s2(L)
            * ((1 / np.tanh(self.optical_depth2(L))) - 1)
            * np.exp(-self.mu2(L) * z)
        )

    # @property
    # def downwelling(self):
    #     def piecewise(z, L):
    #         output = np.empty_like(z)
    #         is_region_1 = z >= -self.thickness_ratio * self.ice_thickness
    #         output[is_region_1] = self._downwelling_1(z[is_region_1], L)
    #         output[~is_region_1] = self._downwelling_2(
    #             z[~is_region_1] + self.ice_thickness * self.thickness_ratio, L
    #         )
    #         return output

    #     return piecewise

    @property
    def downwelling(self):
        return make_piecewise(
            self._downwelling_1,
            self._downwelling_2,
            boundary=-self.ice_thickness * self.thickness_ratio,
        )

    @property
    def upwelling(self):
        return make_piecewise(
            self._upwelling_1,
            self._upwelling_2,
            boundary=-self.ice_thickness * self.thickness_ratio,
        )

    #     @property
    #     def upwelling(self):
    #         def piecewise(z, L):
    #             output = np.empty_like(z)
    #             is_region_1 = z >= -self.thickness_ratio * self.ice_thickness
    #             output[is_region_1] = self._upwelling_1(z[is_region_1], L)
    #             output[~is_region_1] = self._upwelling_2(
    #                 z[~is_region_1] + self.ice_thickness * self.thickness_ratio, L
    #             )
    #             return output

    #         return piecewise

    @property
    def k_cts(self):
        def piecewise(z, L):
            output = np.empty_like(z)
            is_region_1 = z >= -self.thickness_ratio * self.ice_thickness
            output[is_region_1] = self.k1(L)
            output[~is_region_1] = self.k2(L)
            return output

        return piecewise

    @property
    def heating(self):
        return lambda z, L: self.k_cts(z, L) * (
            self.upwelling(z, L) + self.downwelling(z, L)
        )


def make_piecewise(func1, func2, boundary):
    def piecewise_function(z, L):
        output = np.empty_like(z)
        is_region_1 = z >= boundary
        output[is_region_1] = func1(z[is_region_1], L)
        output[~is_region_1] = func2(z[~is_region_1] - boundary, L)
        return output

    return piecewise_function
