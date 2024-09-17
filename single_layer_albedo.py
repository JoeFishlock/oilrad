import numpy as np
import matplotlib.pyplot as plt
from oilrad import two_stream_model

wavelengths = np.linspace(350, 750, 1000)
DROPLET_RADIUS = 0.5

# show two stream optically thick spectral albedo
plt.figure()
for mass_ratio in [0, 1, 10, 100, 200, 1000]:
    model = two_stream_model(
        "1L",
        oil_mass_ratio=mass_ratio,
        ice_thickness=1,
        ice_type="FYI",
        median_droplet_radius_in_microns=DROPLET_RADIUS,
    )
    plt.plot(
        wavelengths,
        model.optically_thick_albedo(wavelengths),
        label=f"{mass_ratio} ng oil/g ice in FYI",
    )
plt.ylim([0, 1])
plt.legend()
plt.grid(True)
plt.xlabel("wavelength (nm)")
plt.ylabel("spectral albedo (optically thick)")
plt.savefig("figures/single_layer/optically_thick_spectral_albedo.pdf")
plt.close()

# Alebdo with thickness included against mass ratio of oil
ice_thickness = 0.8
plt.figure()
plt.title(f"Spectral albedo of FYI {ice_thickness}m thick")
for mass_ratio in [0, 1, 10, 100, 200, 1000]:
    model = two_stream_model(
        "1L",
        oil_mass_ratio=mass_ratio,
        ice_thickness=ice_thickness,
        ice_type="FYI",
        median_droplet_radius_in_microns=DROPLET_RADIUS,
    )
    plt.plot(wavelengths, model.albedo(wavelengths), label=f"{mass_ratio} ng oil/g ice")
plt.ylim([0, 1])
plt.legend()
plt.grid(True)
plt.xlabel("wavelength (nm)")
plt.ylabel("albedo")
plt.savefig("figures/single_layer/spectral_albedo_realistic_thickness.pdf")
plt.close()

# Alebdo against thickness
plt.figure()
plt.title(f"Spectral albedo of FYI")
for ice_thickness in np.linspace(0.5, 6, 6):
    no_oil_model = two_stream_model(
        "1L",
        oil_mass_ratio=0,
        ice_thickness=ice_thickness,
        ice_type="FYI",
        median_droplet_radius_in_microns=DROPLET_RADIUS,
    )
    plt.plot(
        wavelengths,
        no_oil_model.albedo(wavelengths),
        label=f"pure ice {ice_thickness:.1f}m",
    )
for ice_thickness in np.linspace(0.5, 6, 6):
    oil_model = two_stream_model(
        "1L",
        oil_mass_ratio=200,
        ice_thickness=ice_thickness,
        ice_type="FYI",
        median_droplet_radius_in_microns=DROPLET_RADIUS,
    )
    plt.plot(
        wavelengths,
        oil_model.albedo(wavelengths),
        ls="--",
        label=f"200ng oil/g ice {ice_thickness:.1f}m",
    )
plt.ylim([0, 1])
plt.legend(fontsize=5)
plt.grid(True)
plt.xlabel("wavelength (nm)")
plt.ylabel("albedo")
plt.savefig("figures/single_layer/spectral_albedo_against_thickness.pdf")
plt.close()

# Albedo against thickness at 400nm
plt.figure()
plt.title(f"Spectral albedo of FYI at 400nm")

ice_depths = np.linspace(0.5, 15, 100)
no_oil_albedo = []
oil_albedo = []
for ice_thickness in ice_depths:
    no_oil_model = two_stream_model(
        "1L",
        oil_mass_ratio=0,
        ice_thickness=ice_thickness,
        ice_type="FYI",
        median_droplet_radius_in_microns=DROPLET_RADIUS,
    )
    oil_model = two_stream_model(
        "1L",
        oil_mass_ratio=1000,
        ice_thickness=ice_thickness,
        ice_type="FYI",
        median_droplet_radius_in_microns=DROPLET_RADIUS,
    )
    no_oil_albedo.append(no_oil_model.albedo(400))
    oil_albedo.append(oil_model.albedo(400))

plt.plot(ice_depths, no_oil_albedo, label=f"pure ice")
plt.plot(ice_depths, oil_albedo, ls="--", label=f"1000ng oil/g ice")
plt.ylim([0, 1])
plt.legend()
plt.grid(True)
plt.xlabel("ice depth (m)")
plt.ylabel("albedo")
plt.savefig("figures/single_layer/spectral_albedo_against_thickness_at_400nm.pdf")
plt.close()
