import numpy as np
from scipy.integrate import trapezoid
from . import two_stream_model
from .black_body import normalised_black_body_spectrum


def _shortwave_heating_response(
    z, model_choice, min_wavelength, max_wavelength, **kwargs
):
    """Return the heating response of the ice layer weighted by the black body
    spectrum

    Feltham paper says radiation above 700nm is absorped effectively and should
    therefore be included in surface energy balance.

    Function of dimensional depth in ice.
    Provide two stream radiation model choice and the parameters required for it.
    Radiative heating effect is integrated in the given wavelength range.

    NUM SAMPLES sets how many points in wavelength space to take for integration,
    a low number set for efficiency.
    """
    NUM_SAMPLES = 5
    model = two_stream_model(model_choice, **kwargs)
    wavelengths = np.linspace(min_wavelength, max_wavelength, NUM_SAMPLES)
    integrand = np.array(
        [model.heating(z, L) * normalised_black_body_spectrum(L) for L in wavelengths]
    )
    return trapezoid(np.nan_to_num(integrand), wavelengths)


def _shortwave_heating_response_array(
    z_array, model_choice, min_wavelength, max_wavelength, **kwargs
):
    return np.array(
        [
            _shortwave_heating_response(
                z, model_choice, min_wavelength, max_wavelength, **kwargs
            )
            for z in z_array
        ]
    )


def calculate_SW_heating_in_ice(
    incident_shortwave_radiation,
    z_array,
    model_choice,
    min_wavelength,
    max_wavelength,
    **kwargs,
):
    """Given incident shortwave radiation integrated over the full shortwave range
    hitting the ice surface in W/m2 calculate the heating term within the ice at
    the positions in z_array using the radiation model specified.

    The heating term is integrated over the wavelength range specified using a
    black body spectral shape.
    """
    return incident_shortwave_radiation * _shortwave_heating_response_array(
        z_array, model_choice, min_wavelength, max_wavelength, **kwargs
    )
