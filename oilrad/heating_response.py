import numpy as np
from . import two_stream_model
from .black_body import top_of_atmosphere_irradiance, TOTAL_TOP_OF_ATMOSPHERE_IRRADIANCE
from scipy.integrate import trapezoid


def calculate_normalised_incident_shortwave(incident_shortwave_radiation):
    """Incident shortwave given in W/m2 is assumed to be integrated over the entire
    solar top of atmosphere blackbody spectrum"""
    return incident_shortwave_radiation / TOTAL_TOP_OF_ATMOSPHERE_IRRADIANCE


def shortwave_heating_response(
    z, model_choice, min_wavelength, max_wavelength, **kwargs
):
    """Return the heating response of the ice layer to the solar top of atmosphere
    irradiance spectrum integrated over wavelength

    Feltham paper says radiation above 700nm is absorped effectively and should
    therefore be included in surface energy balance.

    Function of dimensional depth in ice.
    Provide two stream radiation model choice and the parameters required for it.
    Radiative heating effect is integrated in the given wavelength range.

    NUM SAMPLES sets how many points in wavelength space to take for integration,
    a low number set for efficiency.
    """
    NUM_SAMPLES = 20
    model = two_stream_model(model_choice, **kwargs)
    wavelengths = np.linspace(min_wavelength, max_wavelength, NUM_SAMPLES)
    integrand = np.array(
        [model.heating(z, L) * top_of_atmosphere_irradiance(L) for L in wavelengths]
    )
    return trapezoid(np.nan_to_num(integrand), wavelengths)


def shortwave_heating_response_array(
    z_array, model_choice, min_wavelength, max_wavelength, **kwargs
):
    return np.array(
        [
            shortwave_heating_response(
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
    return calculate_normalised_incident_shortwave(
        incident_shortwave_radiation
    ) * shortwave_heating_response_array(
        z_array, model_choice, min_wavelength, max_wavelength, **kwargs
    )
