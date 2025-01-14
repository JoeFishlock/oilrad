"""Integration tests for creating a two-stream model configuration and solving for
spectrally integrated output for a variety of input parameters.
"""
import pytest
import numpy as np
import oilrad as oi

Z = np.linspace(-1.5, 0, 1000)
ICE_DEPTH = 0.8
WAVELENGTHS = np.geomspace(350, 3000, 50)
ICE_SCATTERING_COEFFICIENT = 1.5

# set up a variety of model configurations to test
only_ice = oi.CtsWavelengthModel(
    z=Z,
    wavelengths=WAVELENGTHS,
    oil_mass_ratio=np.full_like(Z, 0),
    ice_scattering_coefficient=ICE_SCATTERING_COEFFICIENT,
    median_droplet_radius_in_microns=0.5,
    liquid_fraction=np.full_like(Z, 0),
)

only_water = oi.CtsWavelengthModel(
    z=Z,
    wavelengths=WAVELENGTHS,
    oil_mass_ratio=np.full_like(Z, 0),
    ice_scattering_coefficient=ICE_SCATTERING_COEFFICIENT,
    median_droplet_radius_in_microns=0.5,
    liquid_fraction=np.full_like(Z, 1),
)

no_oil = oi.CtsWavelengthModel(
    z=Z,
    wavelengths=WAVELENGTHS,
    oil_mass_ratio=np.full_like(Z, 0),
    ice_scattering_coefficient=ICE_SCATTERING_COEFFICIENT,
    median_droplet_radius_in_microns=0.5,
    liquid_fraction=np.where(Z >= -ICE_DEPTH, 0, 1),
)
oil_1000 = oi.CtsWavelengthModel(
    z=Z,
    wavelengths=WAVELENGTHS,
    oil_mass_ratio=np.full_like(Z, 1000),
    ice_scattering_coefficient=ICE_SCATTERING_COEFFICIENT,
    median_droplet_radius_in_microns=0.5,
    liquid_fraction=np.where(Z >= -ICE_DEPTH, 0, 1),
)
oil_1000_enhanced = oi.CtsWavelengthModel(
    z=Z,
    wavelengths=WAVELENGTHS,
    oil_mass_ratio=np.full_like(Z, 1000),
    ice_scattering_coefficient=ICE_SCATTERING_COEFFICIENT,
    median_droplet_radius_in_microns=0.5,
    liquid_fraction=np.where(Z >= -ICE_DEPTH, 0, 1),
    absorption_enhancement_factor=1.83,
)


@pytest.mark.parametrize(
    "model", [only_ice, only_water, no_oil, oil_1000, oil_1000_enhanced]
)
def test_cts_wavelength_model(model) -> None:
    """test that solving the two-stream model and integrating the result over the
    blackbody spectrum does not raise an error"""
    oi.integrate_over_SW(
        oi.solve_two_stream_model(model), oi.BlackBodySpectrum(350, 3000)
    )


def test_six_band_model() -> None:
    """test solving the six band model and integrating doesn't throw an error"""
    z = np.linspace(-1, 0, 1000)
    model = oi.SixBandModel(
        z,
        np.full_like(z, 1000),
        ice_scattering_coefficient=800 * 0.15 * 0.75144 * 2,
        median_droplet_radius_in_microns=0.05,
        absorption_enhancement_factor=2,
        snow_depth=0.2,
        snow_spectral_albedos=oi.SNOW_ALBEDOS["light2022"],
        snow_extinction_coefficients=oi.SNOW_EXTINCTION_COEFFICIENTS["lebrun2023"],
        SSL_depth=0.04,
        SSL_spectral_albedos=oi.SSL_ALBEDOS["light2022"],
        SSL_extinction_coefficients=oi.SSL_EXTINCTION_COEFFICIENTS["perovich1990"],
    )
    oi.integrate_over_SW(oi.solve_two_stream_model(model))


def test_cts_wavelength_fast_solve() -> None:
    """Test that the fast solver agrees with the full solver below the wavelength cutoff"""
    WAVELENGTH_CUTOFF = 1200
    full_model = oi.CtsWavelengthModel(
        z=Z,
        wavelengths=WAVELENGTHS,
        oil_mass_ratio=np.full_like(Z, 0),
        ice_scattering_coefficient=ICE_SCATTERING_COEFFICIENT,
        median_droplet_radius_in_microns=0.5,
        liquid_fraction=np.where(Z >= -ICE_DEPTH, 0, 1),
    )
    fast_model = oi.CtsWavelengthModel(
        z=Z,
        wavelengths=WAVELENGTHS,
        oil_mass_ratio=np.full_like(Z, 0),
        ice_scattering_coefficient=ICE_SCATTERING_COEFFICIENT,
        median_droplet_radius_in_microns=0.5,
        liquid_fraction=np.where(Z >= -ICE_DEPTH, 0, 1),
        fast_solve=True,
        wavelength_cutoff=WAVELENGTH_CUTOFF,
    )
    full_solution = oi.solve_two_stream_model(full_model)
    fast_solution = oi.solve_two_stream_model(fast_model)
    short_wavelengths = WAVELENGTHS < WAVELENGTH_CUTOFF
    assert np.all(
        full_solution.albedo[short_wavelengths]
        == fast_solution.albedo[short_wavelengths]
    )
