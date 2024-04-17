"""Estimate oil absorption in ice neglecting mie calculation

Just use data for oil absorption from Otremba 2007 Table 1
"""

import numpy as np
import matplotlib.pyplot as plt
from gulf.optics import (
    calculate_ice_absorption_coefficient,
    calculate_ice_oil_absorption_coefficient,
    calculate_ice_oil_extinction_coefficient,
)


wavelengths = np.linspace(350, 750, 1000)


# Show pure ice absorption and oil absorption for different mass ratios
plt.figure()
plt.title("Effect of oil pollution on ice spectral absorption")
for mass_ratio in [0, 1, 10, 100, 1000]:
    plt.plot(
        wavelengths,
        calculate_ice_oil_absorption_coefficient(wavelengths, mass_ratio),
        label=f"{mass_ratio} ng oil/g ice",
    )
plt.plot(
    wavelengths,
    calculate_ice_absorption_coefficient(wavelengths),
    label="pure ice absorption",
)
plt.yscale("log")
plt.xlabel("wavelength (nm)")
plt.xlim([350, 750])
plt.ylim([1e-4, 1e1])
plt.ylabel("absorption coefficient (1/m)")
plt.grid(True)
plt.savefig("figures/Romashkino_oil_absorption.pdf")
plt.close()

# Show extinction coefficient from two stream model for different oil concs
plt.figure()
for mass_ratio in [0, 1, 10, 100, 1000]:
    plt.plot(
        wavelengths,
        calculate_ice_oil_extinction_coefficient(
            wavelengths, mass_ratio, ice_type="FYI"
        ),
        label=f"{mass_ratio} ng oil/g ice in FYI",
    )

plt.yscale("log")
plt.ylabel("extinction coefficient (1/m)")
plt.xlabel("wavelength (nm)")
plt.legend()
plt.grid(True)
plt.savefig("figures/Otremba_oil_extinction_coefficient.pdf")
