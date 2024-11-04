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
only_ice = oi.InfiniteLayerModel(
    z=Z,
    wavelengths=WAVELENGTHS,
    oil_mass_ratio=np.full_like(Z, 0),
    ice_scattering_coefficient=ICE_SCATTERING_COEFFICIENT,
    median_droplet_radius_in_microns=0.5,
    liquid_fraction=np.full_like(Z, 0),
)

only_water = oi.InfiniteLayerModel(
    z=Z,
    wavelengths=WAVELENGTHS,
    oil_mass_ratio=np.full_like(Z, 0),
    ice_scattering_coefficient=ICE_SCATTERING_COEFFICIENT,
    median_droplet_radius_in_microns=0.5,
    liquid_fraction=np.full_like(Z, 1),
)

no_oil = oi.InfiniteLayerModel(
    z=Z,
    wavelengths=WAVELENGTHS,
    oil_mass_ratio=np.full_like(Z, 0),
    ice_scattering_coefficient=ICE_SCATTERING_COEFFICIENT,
    median_droplet_radius_in_microns=0.5,
    liquid_fraction=np.where(Z >= -ICE_DEPTH, 0, 1),
)
oil_1000 = oi.InfiniteLayerModel(
    z=Z,
    wavelengths=WAVELENGTHS,
    oil_mass_ratio=np.full_like(Z, 1000),
    ice_scattering_coefficient=ICE_SCATTERING_COEFFICIENT,
    median_droplet_radius_in_microns=0.5,
    liquid_fraction=np.where(Z >= -ICE_DEPTH, 0, 1),
)
oil_1000_enhanced = oi.InfiniteLayerModel(
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
def test_solve(model) -> None:
    """test that solving the two-stream model and integrating the result over the
    blackbody spectrum does not raise an error"""
    oi.integrate_over_SW(
        oi.solve_two_stream_model(model), oi.BlackBodySpectrum(350, 3000)
    )


def test_fast_solve() -> None:
    """Test that the fast solver agrees with the full solver below the wavelength cutoff"""
    WAVELENGTH_CUTOFF = 1200
    full_model = oi.InfiniteLayerModel(
        z=Z,
        wavelengths=WAVELENGTHS,
        oil_mass_ratio=np.full_like(Z, 0),
        ice_scattering_coefficient=ICE_SCATTERING_COEFFICIENT,
        median_droplet_radius_in_microns=0.5,
        liquid_fraction=np.where(Z >= -ICE_DEPTH, 0, 1),
    )
    fast_model = oi.InfiniteLayerModel(
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
