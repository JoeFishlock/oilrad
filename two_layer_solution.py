import numpy as np
import matplotlib.pyplot as plt
from oilrad.two_layer import PAR_heating
from oilrad import two_stream_model
from oilrad.black_body import solar_irradiance
from scipy.integrate import quad

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
unpolluted_albedo = two_stream_model(
    "1L", oil_mass_ratio=0, ice_thickness=ICE_THICKNESS, ice_type=ICE_TYPE
).albedo(wavelengths)
for thickness_ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
    model = two_stream_model(
        "2L",
        oil_mass_ratio=OIL,
        thickness_ratio=thickness_ratio,
        ice_thickness=ICE_THICKNESS,
        ice_type=ICE_TYPE,
    )
    plt.plot(
        wavelengths,
        model.albedo(wavelengths),
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
base_albedo = two_stream_model(
    "1L", oil_mass_ratio=OIL, ice_thickness=ICE_THICKNESS, ice_type=ICE_TYPE
).albedo(wavelengths)
for thickness_ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
    model = two_stream_model(
        "2L",
        oil_mass_ratio=OIL,
        thickness_ratio=thickness_ratio,
        ice_thickness=ICE_THICKNESS,
        ice_type=ICE_TYPE,
    )
    difference_percentage = (
        100 * (model.albedo(wavelengths) - base_albedo) / base_albedo
    )
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
    base_albedo = two_stream_model(
        "1L", oil_mass_ratio=OIL, ice_thickness=ICE_THICKNESS, ice_type=ICE_TYPE
    ).albedo(wavelength)
    unpolluted_albedo = two_stream_model(
        "1L", oil_mass_ratio=0, ice_thickness=ICE_THICKNESS, ice_type=ICE_TYPE
    ).albedo(wavelength)
    plt.plot(1, unpolluted_albedo, light_color, marker="o", alpha=0.3)
    plt.plot(1, base_albedo, light_color, marker="*")

    modeled_albedos = []
    for thickness_ratio in f:
        model = two_stream_model(
            "2L",
            oil_mass_ratio=OIL,
            thickness_ratio=thickness_ratio,
            ice_thickness=ICE_THICKNESS,
            ice_type=ICE_TYPE,
        )
        modeled_albedos.append(model.albedo(wavelength))

    plt.plot(
        f,
        np.array(modeled_albedos),
        light_color,
        label=f"{wavelength}nm",
    )

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

model = two_stream_model(
    "2L", oil_mass_ratio=oil, thickness_ratio=f, ice_thickness=h, ice_type=ice_type
)
no_oil = two_stream_model(
    "2L", oil_mass_ratio=0, thickness_ratio=f, ice_thickness=h, ice_type=ice_type
)
uniform = two_stream_model(
    "2L", oil_mass_ratio=oil, thickness_ratio=0.99, ice_thickness=h, ice_type=ice_type
)


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

# Integrate over PAR range using solar irradiance blackbody
z = np.linspace(-ICE_THICKNESS, 0, 15)
solar_irradiance_func = lambda L: solar_irradiance(L, environment_conditions=3)
incident_PAR_flux = quad(solar_irradiance_func, 350, 700)[0]

# Have to manually vectorize the function here as using boolean indexing in the
# TwoLayerModel class
oil_PAR_heating = [
    PAR_heating(model, solar_irradiance=solar_irradiance_func)(np.array([depth]))
    for depth in z
]
print("1/3 heating profiles integrated 350nm-700nm")
no_oil_PAR_heating = [
    PAR_heating(no_oil, solar_irradiance=solar_irradiance_func)(np.array([depth]))
    for depth in z
]
print("2/3 heating profiles integrated 350nm-700nm")
uniform_oil_PAR_heating = [
    PAR_heating(uniform, solar_irradiance=solar_irradiance_func)(np.array([depth]))
    for depth in z
]
print("3/3 heating profiles integrated 350nm-700nm")
plt.figure()
plt.title(
    f"350nm-700nm heating {ice_type} {h}m thick, incident DSW {incident_PAR_flux:.1f}W/m2"
)
plt.plot(oil_PAR_heating, z, "ro--", label=f"2layer {oil}ng oil/g ice")
plt.plot(no_oil_PAR_heating, z, "bo--", label="no oil")
plt.plot(uniform_oil_PAR_heating, z, "go--", label=f"uniform {oil}ng oil/g ice")
plt.axhline(y=-f * h, ls=":", color="k", label="oil layer boundary")
plt.xlabel("wavelength integrated radiative heating W/m3")
plt.ylabel("depth (m)")
plt.legend()
plt.grid(True)
plt.savefig("figures/two_layer/two_layer_PAR_heating.pdf")
