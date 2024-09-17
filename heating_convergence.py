"""Examine convergence of discrete geometric sampling in wavelength space for heating integration
"""
import numpy as np
import matplotlib.pyplot as plt
import oilrad

min_wavelength = 350
max_wavelength = 1200
ice_thickness = 1
ice_type = "FYI"
oil_mass_ratio = 0
median_droplet_radius = 0.5
max_samples = 20
FSI = 280

model = oilrad.two_stream_model(
    "1L",
    oil_mass_ratio=oil_mass_ratio,
    ice_thickness=ice_thickness,
    ice_type=ice_type,
    median_droplet_radius_in_microns=median_droplet_radius,
)
truth = oilrad.calculate_SW_heating_in_ice(
    FSI,
    np.linspace(-ice_thickness, 0, 5),
    "1L",
    min_wavelength,
    max_wavelength,
    num_samples=max_samples,
    ice_thickness=ice_thickness,
    ice_type=ice_type,
    oil_mass_ratio=oil_mass_ratio,
    median_droplet_radius_in_microns=median_droplet_radius,
)
plt.figure()
for num_samples in range(2, max_samples + 1):
    heating = oilrad.calculate_SW_heating_in_ice(
        FSI,
        np.linspace(-ice_thickness, 0, 5),
        "1L",
        min_wavelength,
        max_wavelength,
        num_samples=num_samples,
        ice_thickness=ice_thickness,
        ice_type=ice_type,
        oil_mass_ratio=oil_mass_ratio,
        median_droplet_radius_in_microns=median_droplet_radius,
    )
    # mean of relative error
    error = np.mean(np.abs((heating - truth) / truth))
    plt.plot(float(num_samples), error, "k*")

plt.yscale("log")
plt.xscale("log")
plt.xlabel("number of samples in wavelength space")
plt.ylabel("mean relative error in heating profile")
plt.title("convergence of heating integrator")
plt.savefig("figures/convergence_of_heating_integration.pdf")
