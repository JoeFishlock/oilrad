"""Solve the model with continuously varying optical parameters"""

import numpy as np
from gulf.optics import (
    calculate_ice_oil_absorption_coefficient,
    calculate_ice_scattering_coefficient_from_Roche_2022,
)
from dataclasses import dataclass
from typing import Callable
from scipy.integrate import solve_bvp


@dataclass
class InfiniteLayerModel:
    """F = [upwelling(z, L), downwelling(z, L)]"""

    oil_mass_ratio: Callable[[float], float]
    ice_thickness: float
    ice_type: str

    @property
    def r(self):
        return calculate_ice_scattering_coefficient_from_Roche_2022(self.ice_type)

    @property
    def k(self):
        return lambda z, L: calculate_ice_oil_absorption_coefficient(
            L, oil_mass_ratio=self.oil_mass_ratio(z)
        )

    def get_ODE_fun(self, L):
        upwelling_part = lambda z, F: -(self.k(z, L) + self.r) * F[0] + self.r * F[1]
        downwelling_part = lambda z, F: (self.k(z, L) + self.r) * F[1] - self.r * F[0]
        return lambda z, F: np.vstack((upwelling_part(z, F), downwelling_part(z, F)))

    @property
    def BCs(self):
        """Doesn't depend on wavelength"""
        return lambda F_bottom, F_top: np.array([F_top[1] - 1, F_bottom[0]])

    def _get_system_solution(self, L):
        ODE_fun = self.get_ODE_fun(L)
        solution = solve_bvp(
            ODE_fun, self.BCs, np.linspace(-self.ice_thickness, 0, 5), np.zeros((2, 5))
        ).sol
        return solution

    @property
    def get_upwelling_and_downwelling(self):
        upwelling = lambda z, L: self._get_system_solution(L)(z)[0]
        downwelling = lambda z, L: self._get_system_solution(L)(z)[1]
        return upwelling, downwelling

    @property
    def albedo(self):
        return lambda L: self.get_upwelling_and_downwelling[0](0, L)

    @property
    def downwelling(self):
        pass

    @property
    def upwelling(self):
        pass

    @property
    def heating(self):
        pass
