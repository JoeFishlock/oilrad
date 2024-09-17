"""Compare discrete geometric sampling in wavelength space for albedo integration
to quad integrator
"""
from scipy.integrate import quad, trapezoid
import numpy as np
import matplotlib.pyplot as plt
import oilrad
from oilrad.black_body import normalised_black_body_spectrum

min_wavelength = 350
max_wavelength = 1200
ice_thickness = 1
ice_type = "FYI"
oil_mass_ratio = 0
median_droplet_radius = 0.5

model = oilrad.two_stream_model(
    "1L",
    oil_mass_ratio=oil_mass_ratio,
    ice_thickness=ice_thickness,
    ice_type=ice_type,
    median_droplet_radius_in_microns=median_droplet_radius,
)
quad_integrator = quad(
    lambda L: model.albedo(L) * normalised_black_body_spectrum(L),
    min_wavelength,
    max_wavelength,
)[0]
plt.figure()
for num_samples in range(2, 51):
    waves = np.geomspace(min_wavelength, max_wavelength, num_samples)
    error = (
        trapezoid(normalised_black_body_spectrum(waves) * model.albedo(waves), waves)
        - quad_integrator
    )
    plt.plot(float(num_samples), np.abs(error), "k*")

plt.yscale("log")
plt.xscale("log")
plt.xlabel("number of samples in wavelength space")
plt.ylabel("error in total albedo integral")
plt.title("convergence of total albedo integrator")
plt.savefig("figures/convergence_of_albedo_integration.pdf")
