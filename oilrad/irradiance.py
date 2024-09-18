"""Classes to store solution of two stream spectral model and integrate over
the SW spectrum
"""

from dataclasses import dataclass
from numpy.typing import NDArray
from scipy.integrate import trapezoid
from .black_body import get_normalised_black_body_spectrum


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
    spectrum = get_normalised_black_body_spectrum((wavelengths[0], wavelengths[-1]))
    integrate = lambda irradiance: trapezoid(
        irradiance * spectrum(wavelengths), wavelengths, axis=1
    )
    integrated_upwelling = integrate(spectral_irradiance.upwelling)
    integrated_downwelling = integrate(spectral_irradiance.downwelling)
    return Irradiance(
        spectral_irradiance.z, integrated_upwelling, integrated_downwelling
    )
