import numpy as np
import matplotlib.pyplot as plt
from oilrad import two_stream_model

plt.style.use(["science", "nature", "grid"])

ICE_THICKNESS = 1
ICE_TYPE = "FYI"
WAVELENGTH = 400
OIL = 1000
DROPLET_RADIUS = 0.5

no_oil_model = two_stream_model(
    "1L",
    oil_mass_ratio=0,
    ice_thickness=ICE_THICKNESS,
    ice_type=ICE_TYPE,
    median_droplet_radius_in_microns=DROPLET_RADIUS,
)
oil_model = two_stream_model(
    "1L",
    oil_mass_ratio=OIL,
    ice_thickness=ICE_THICKNESS,
    ice_type=ICE_TYPE,
    median_droplet_radius_in_microns=DROPLET_RADIUS,
)

z = np.linspace(-ICE_THICKNESS, 0, 100)

plt.figure(figsize=(4, 3))
plt.title(f"Wavelength {WAVELENGTH}nm irradiance in {ICE_TYPE} {ICE_THICKNESS}m thick")
plt.xlabel("Planar spectral irradiance / incident solar irradiance")
plt.ylabel("Depth (m)")
plt.plot(
    no_oil_model.upwelling(z, WAVELENGTH),
    z,
    "b",
    label="upwelling no oil",
)
plt.plot(
    oil_model.upwelling(z, WAVELENGTH),
    z,
    "r",
    label=f"upwelling {OIL}ng oil/g ice",
)
plt.plot(
    no_oil_model.downwelling(z, WAVELENGTH),
    z,
    "b--",
    label="downwelling no oil",
)
plt.plot(
    oil_model.downwelling(z, WAVELENGTH),
    z,
    "r--",
    label=f"downwelling {OIL}ng oil/g ice",
)
plt.plot(
    no_oil_model.net_radiation(z, WAVELENGTH),
    z,
    "b:",
    label="net rad no oil",
)
plt.plot(
    oil_model.net_radiation(z, WAVELENGTH),
    z,
    "r:",
    label=f"net rad {OIL}ng oil/g ice",
)
plt.legend()
plt.savefig("figures/single_layer/radiation_streams.pdf")
plt.close()

"""This figure shows the radiative heating for several oil concentrations

The heating must be integrated over the incident wavelength spectrum
"""
plt.figure(figsize=(4, 3))
plt.title(f"Radiative heating at {WAVELENGTH}nm in {ICE_TYPE} {ICE_THICKNESS}m thick")
plt.xlabel("Radiative heating / incident solar irradiance")
plt.ylabel("Depth (m)")
for oil_mass_ratio in [0, 1, 10, 100, 1000]:
    model = two_stream_model(
        "1L",
        oil_mass_ratio=oil_mass_ratio,
        ice_thickness=ICE_THICKNESS,
        ice_type=ICE_TYPE,
        median_droplet_radius_in_microns=DROPLET_RADIUS,
    )
    plt.plot(
        model.heating(z, WAVELENGTH),
        z,
        label=f"heating {oil_mass_ratio} ng oil/g ice",
    )
plt.legend()
plt.savefig("figures/single_layer/radiative_heating.pdf")
plt.close()

"""This figure shows the amount of shortwave radiation absorbed by the ice is many times
greater even at realistic thickness and moderate oil concentrations"""
plt.figure(figsize=(4, 3))
plt.title(
    f"Radiative heating change at {WAVELENGTH}nm in {ICE_TYPE} {ICE_THICKNESS}m thick"
)
plt.xlabel("Radiative heating / radiative heating no oil (%)")
plt.ylabel("Depth (m)")
base_heating = no_oil_model.heating(z, WAVELENGTH)
for oil_mass_ratio in [0, 1, 10, 100]:
    model = two_stream_model(
        "1L",
        oil_mass_ratio=oil_mass_ratio,
        ice_thickness=ICE_THICKNESS,
        ice_type=ICE_TYPE,
        median_droplet_radius_in_microns=DROPLET_RADIUS,
    )
    plt.plot(
        100 * (model.heating(z, WAVELENGTH) - base_heating) / base_heating,
        z,
        label=f"{oil_mass_ratio} ng oil/g ice",
    )
plt.legend()
plt.savefig("figures/single_layer/radiative_heating_change.pdf")
plt.close()
