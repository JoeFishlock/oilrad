import numpy as np
import matplotlib.pyplot as plt
from gulf.two_layer import calculate_albedo, TwoLayerModel
from gulf.single_layer import calculate_albedo as single_layer_albedo

wavelengths = np.linspace(350, 750, 1000)
OIL = 100
ICE_TYPE = "MYI"
ICE_THICKNESS = 2

# Alebdo for different oil layer thickness ratios
plt.figure(figsize=(8, 6))
ax = plt.gca()
ax.set_facecolor("#7c7afc")
plt.title(
    f"Two layer spectral albedo of {ICE_TYPE} {ICE_THICKNESS}m thick with {OIL}ng oil/g ice"
)
unpolluted_albedo = single_layer_albedo(ICE_THICKNESS, wavelengths, 0, ICE_TYPE)
for thickness_ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
    albedo = calculate_albedo(
        ICE_THICKNESS,
        wavelengths,
        oil_mass_ratio=OIL,
        ice_type=ICE_TYPE,
        thickness_ratio=thickness_ratio,
    )
    plt.plot(
        wavelengths,
        albedo,
        color=str(thickness_ratio),
        label=f"thickness ratio {thickness_ratio}",
    )
plt.plot(wavelengths, unpolluted_albedo, "r--", label="unpolluted")
# plt.ylim([0, 1])
plt.legend(frameon=False)
# plt.grid(True)
plt.xlabel("wavelength (nm)")
plt.ylabel("albedo")
plt.savefig("figures/two_layer/spectral_albedo_oil_layer.pdf")
plt.close()

# Alebdo change for different oil layer thickness ratios
plt.figure(figsize=(8, 6))
ax = plt.gca()
ax.set_facecolor("#7c7afc")
plt.title(
    f"Two layer spectral albedo change of {ICE_TYPE} {ICE_THICKNESS}m thick with {OIL}ng oil/g ice"
)
base_albedo = single_layer_albedo(ICE_THICKNESS, wavelengths, OIL, ICE_TYPE)
for thickness_ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
    albedo = calculate_albedo(
        ICE_THICKNESS,
        wavelengths,
        oil_mass_ratio=OIL,
        ice_type=ICE_TYPE,
        thickness_ratio=thickness_ratio,
    )
    difference_percentage = 100 * (albedo - base_albedo) / base_albedo
    plt.plot(
        wavelengths,
        difference_percentage,
        color=str(thickness_ratio),
        label=f"thickness ratio {thickness_ratio}",
    )
# plt.ylim([0, 1])
plt.legend(frameon=False)
# plt.grid(True)
plt.xlabel("wavelength (nm)")
plt.ylabel("albedo change (%)")
plt.savefig("figures/two_layer/spectral_albedo_oil_layer_difference.pdf")
plt.close()

# Alebdo for different oil layer thickness ratios at specific wavelengths
plt.figure(figsize=(8, 6))
plt.title(
    f"Two layer spectral albedo of {ICE_TYPE} {ICE_THICKNESS}m thick with {OIL}ng oil/g ice"
)

discrete_wavelengths = [350, 400, 450, 550, 600]
light_colors = ["m--", "m", "b", "g", "y"]
f = np.linspace(0.01, 0.99, 200)
for wavelength, light_color in zip(discrete_wavelengths, light_colors):
    albedo = calculate_albedo(
        ICE_THICKNESS,
        wavelength,
        oil_mass_ratio=OIL,
        ice_type=ICE_TYPE,
        thickness_ratio=f,
    )
    base_albedo = single_layer_albedo(ICE_THICKNESS, wavelength, OIL, ICE_TYPE)
    unpolluted_albedo = single_layer_albedo(ICE_THICKNESS, wavelength, 0, ICE_TYPE)
    plt.plot(1, unpolluted_albedo, light_color, marker="o", alpha=0.3)
    plt.plot(f, albedo, light_color, label=f"{wavelength}nm")
    plt.plot(1, base_albedo, light_color, marker="*")

# plt.ylim([0, 1])
plt.legend(frameon=False)
plt.grid(True)
plt.xlabel("oil layer thickness ratio")
plt.ylabel("spectral albedo")
plt.savefig("figures/two_layer/oil_layer_albedos.pdf")
plt.close()

"""Two layer model solution

First you define model parameters and pass them to the TwoLayerModel class

from this you can extract functions that give the spectral albedo, upwelling irradiance
downwelling irradiance and radiative heating
"""
oil = 1000
f = 0.2
h = 2
wavelength = 400
ice_type = "FYI"

model = TwoLayerModel(oil, f, h, ice_type)
no_oil = TwoLayerModel(0, f, h, ice_type)


plt.figure()
plt.title(f"Irradiance at {wavelength}nm in {ice_type} {h}m thick")

z = np.linspace(-h, 0, 100)

downwelling = model.downwelling
upwelling = model.upwelling
plt.plot(upwelling(z, wavelength), z, "r--", label=f"upwelling {oil}ng oil/g ice")
plt.plot(downwelling(z, wavelength), z, "r", label=f"downwelling {oil}ng oil/g ice")

downwelling = no_oil.downwelling
upwelling = no_oil.upwelling
plt.plot(upwelling(z, wavelength), z, "b--", label="upwelling pure ice")
plt.plot(downwelling(z, wavelength), z, "b", label="downwelling pure ice")

plt.axhline(y=-f * h, ls=":", color="k", label="oil layer boundary")
plt.xlabel("non dimensional irradiance")
plt.ylabel("depth (m)")
plt.legend()
plt.savefig("figures/two_layer/two_layer_irradiance.pdf")

plt.figure()
plt.title(f"Radiative heating at {wavelength}nm in {ice_type} {h}m thick")
heating_oil = model.heating
heating_no_oil = no_oil.heating
plt.plot(heating_oil(z, wavelength), z, "r", label=f"uniform {oil}ng oil/g ice")
plt.plot(heating_no_oil(z, wavelength), z, "b", label="no oil")
plt.axhline(y=-f * h, ls=":", color="k", label="oil layer boundary")
plt.xlabel("radiative heating")
plt.ylabel("depth (m)")
plt.legend()
plt.savefig("figures/two_layer/two_layer_heating.pdf")
