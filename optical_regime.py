import matplotlib.pyplot as plt
from gulf.optics import calculate_ice_oil_extinction_coefficient
import numpy as np

plt.figure()
plt.title("Optical regime diagram for single layer two stream model")
plt.xlabel("extinction coefficient (1/m)")
plt.ylabel("ice thickness (m)")
plt.xscale("log")
plt.xlim([1e-2, 1e1])
plt.ylim([1e-1, 1e1])
plt.yscale("log")
x_values = np.linspace(1e-3, 1e3, 1000)

plt.plot(x_values, 1 / x_values, "k", label="optical depth=1")

# first year ice
for oil_mass_ratio, line_style in zip([0, 1000], ["-", "--"]):
    for wavelength, color in zip([350, 750], ["violet", "red"]):
        extinction_coefficient = calculate_ice_oil_extinction_coefficient(
            wavelength, oil_mass_ratio, ice_type="FYI"
        )
        plt.vlines(
            extinction_coefficient,
            ymin=0.5,
            ymax=1.5,
            linestyles=line_style,
            colors=color,
        )


plt.legend()
plt.grid(True)
plt.savefig("optical_regime_single_layer.pdf")
