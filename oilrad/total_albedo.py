import numpy as np
from scipy.integrate import trapezoid
from .abstract_model import AbstractModel
from .black_body import normalised_black_body_spectrum


def calculate_total_albedo(
    model: AbstractModel, min_wavelength: float, max_wavelength: float, num_samples=20
) -> float:
    """For given wavelength interbal in nm integrate the modelled spectral albedo
    over a normalised blackbody spectrum

    discrete integration in wavelength space with given number of samples geometrically
    distributed.

    Preliminary tests show that num_samples=20 should be sufficient to guarantee less than
    1e-3 error in the integrated total albedo
    """
    wavelengths = np.geomspace(min_wavelength, max_wavelength, num_samples)
    return trapezoid(
        model.albedo(wavelengths) * normalised_black_body_spectrum(wavelengths),
        wavelengths,
    )
