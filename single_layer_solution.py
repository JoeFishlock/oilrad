import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

from gulf.single_layer import (
    calculate_ND_downwelling,
    calculate_ND_upwelling,
    calculate_albedo,
    calculate_ND_heating,
    calculate_ND_net_radiation,
)
from gulf.optics import calculate_ice_oil_extinction_coefficient

plt.style.use(["science", "nature", "grid"])

ICE_THICKNESS = 1
ICE_TYPE = "FYI"
WAVELENGTH = 400
OIL = 1000

print(calculate_albedo(ICE_THICKNESS, WAVELENGTH, OIL, ICE_TYPE))
print(
    "optical_thickness: ",
    calculate_ice_oil_extinction_coefficient(WAVELENGTH, OIL, ICE_TYPE) * ICE_THICKNESS,
)

z = np.linspace(-ICE_THICKNESS, 0, 100)

plt.figure(figsize=(4, 3))
plt.title(f"Wavelength {WAVELENGTH}nm irradiance in {ICE_TYPE} {ICE_THICKNESS}m thick")
plt.xlabel("Planar spectral irradiance / incident solar irradiance")
plt.ylabel("Depth (m)")
plt.plot(
    calculate_ND_upwelling(z, ICE_THICKNESS, WAVELENGTH, 0, ICE_TYPE),
    z,
    "b",
    label="upwelling no oil",
)
plt.plot(
    calculate_ND_upwelling(z, ICE_THICKNESS, WAVELENGTH, OIL, ICE_TYPE),
    z,
    "r",
    label=f"upwelling {OIL}ng oil/g ice",
)
plt.plot(
    calculate_ND_downwelling(z, ICE_THICKNESS, WAVELENGTH, 0, ICE_TYPE),
    z,
    "b--",
    label="downwelling no oil",
)
plt.plot(
    calculate_ND_downwelling(z, ICE_THICKNESS, WAVELENGTH, OIL, ICE_TYPE),
    z,
    "r--",
    label=f"downwelling {OIL}ng oil/g ice",
)
plt.plot(
    calculate_ND_net_radiation(z, ICE_THICKNESS, WAVELENGTH, 0, ICE_TYPE),
    z,
    "b:",
    label="net rad no oil",
)
plt.plot(
    calculate_ND_net_radiation(z, ICE_THICKNESS, WAVELENGTH, OIL, ICE_TYPE),
    z,
    "r:",
    label=f"net rad {OIL}ng oil/g ice",
)
plt.legend()
plt.savefig("figures/radiation_streams.pdf")
plt.close()

"""This figure shows the radiative heating for several oil concentrations

The heating must be integrated over the incident wavelength spectrum
"""
plt.figure(figsize=(4, 3))
plt.title(f"Radiative heating at {WAVELENGTH}nm in {ICE_TYPE} {ICE_THICKNESS}m thick")
plt.xlabel("Radiative heating / incident solar irradiance")
plt.ylabel("Depth (m)")
for oil_mass_ratio in [0, 1, 10, 100, 1000]:
    plt.plot(
        calculate_ND_heating(z, ICE_THICKNESS, WAVELENGTH, oil_mass_ratio, ICE_TYPE),
        z,
        label=f"heating {oil_mass_ratio} ng oil/g ice",
    )
plt.legend()
plt.savefig("figures/radiative_heating.pdf")
plt.close()

"""This figure shows the amount of shortwave radiation absorbed by the ice is many times
greater even at realistic thickness and moderate oil concentrations"""
plt.figure(figsize=(4, 3))
plt.title(
    f"Radiative heating change at {WAVELENGTH}nm in {ICE_TYPE} {ICE_THICKNESS}m thick"
)
plt.xlabel("Radiative heating / radiative heating no oil (%)")
plt.ylabel("Depth (m)")
base_heating = calculate_ND_heating(z, ICE_THICKNESS, WAVELENGTH, 0, ICE_TYPE)
for oil_mass_ratio in [0, 1, 10, 100]:
    plt.plot(
        100
        * (
            calculate_ND_heating(z, ICE_THICKNESS, WAVELENGTH, oil_mass_ratio, ICE_TYPE)
            - base_heating
        )
        / base_heating,
        z,
        label=f"{oil_mass_ratio} ng oil/g ice",
    )
plt.legend()
plt.savefig("figures/radiative_heating_change.pdf")
plt.close()

plt.figure(figsize=(4, 3))
plt.title(
    f"depth integrated radiative heating spectrum in {ICE_TYPE} {ICE_THICKNESS}m thick"
)
plt.ylabel("depth integrated radiative heating / incident solar irradiance spectrum")
plt.xlabel("wavelength (nm)")
spectrum = np.linspace(350, 750, 1000)
for oil_mass_ratio in [0, 1, 10, 100, 1000]:
    broadband_heating = []
    for wavelength in spectrum:
        radiative_heating = lambda z: calculate_ND_heating(
            z, ICE_THICKNESS, wavelength, oil_mass_ratio, ICE_TYPE
        )
        integrated_radiative_heating, _ = quad(radiative_heating, -ICE_THICKNESS, 0)
        broadband_heating.append(integrated_radiative_heating)

    plt.plot(
        spectrum,
        broadband_heating,
        label=f"{oil_mass_ratio} ng oil/g ice",
    )
plt.yscale("log")
plt.legend()
plt.savefig("figures/radiative_heating_depth_integrated.pdf")
plt.close()
