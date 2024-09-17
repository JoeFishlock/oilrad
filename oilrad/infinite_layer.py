"""Solve the model with continuously varying optical parameters"""

import numpy as np
from oilrad.optics import (
    calculate_ice_oil_absorption_coefficient,
    calculate_ice_scattering_coefficient_from_Roche_2022,
)
from oilrad.abstract_model import AbstractModel
from dataclasses import dataclass
from typing import Callable
from scipy.integrate import solve_bvp


@dataclass
class InfiniteLayerModel(AbstractModel):
    """F = [upwelling(z, L), downwelling(z, L)]"""

    oil_mass_ratio: Callable[[float], float]
    ice_thickness: float
    ice_type: str
    median_droplet_radius_in_microns: float

    @property
    def r(self):
        return calculate_ice_scattering_coefficient_from_Roche_2022(self.ice_type)

    def k(self, z, L):
        return calculate_ice_oil_absorption_coefficient(
            L,
            oil_mass_ratio=self.oil_mass_ratio(z),
            droplet_radius_in_microns=self.median_droplet_radius_in_microns,
        )

    def _ODE_fun(self, z, F, L):
        upwelling_part = -(self.k(z, L) + self.r) * F[0] + self.r * F[1]
        downwelling_part = (self.k(z, L) + self.r) * F[1] - self.r * F[0]
        return np.vstack((upwelling_part, downwelling_part))

    def _BCs(self, F_bottom, F_top):
        """Doesn't depend on wavelength"""
        return np.array([F_top[1] - 1, F_bottom[0]])

    def _get_system_solution(self, L):
        solution = solve_bvp(
            lambda z, F: self._ODE_fun(z, F, L=L),
            self._BCs,
            np.linspace(-self.ice_thickness, 0, 5),
            np.zeros((2, 5)),
        ).sol
        return solution

    def upwelling(self, z, L):
        return self._get_system_solution(L)(z)[0]

    def downwelling(self, z, L):
        return self._get_system_solution(L)(z)[1]

    def albedo(self, L):
        return np.vectorize(self.upwelling)(0, L)

    def transmittance(self, L):
        return np.vectorize(self.downwelling)(-self.ice_thickness, L)

    def heating(self, z, L):
        return self.k(z, L) * (self.upwelling(z, L) + self.downwelling(z, L))
