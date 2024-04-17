import numpy as np
import matplotlib.pyplot as plt
from gulf.single_layer import (
    calculate_optically_thick_albedo,
    calculate_albedo,
)

wavelengths = np.linspace(350, 750, 1000)

# show two stream optically thick spectral albedo
plt.figure()
for mass_ratio in [0, 1, 10, 100, 200, 1000]:
    plt.plot(
        wavelengths,
        calculate_optically_thick_albedo(wavelengths, mass_ratio, ice_type="FYI"),
        label=f"{mass_ratio} ng oil/g ice in FYI",
    )
plt.ylim([0, 1])
plt.legend()
plt.grid(True)
plt.xlabel("wavelength (nm)")
plt.ylabel("spectral albedo (optically thick)")
plt.savefig("figures/optically_thick_spectral_albedo.pdf")
plt.close()

# Alebdo with thickness included against mass ratio of oil
ice_thickness = 0.8
plt.figure()
plt.title(f"Spectral albedo of FYI {ice_thickness}m thick")
for mass_ratio in [0, 1, 10, 100, 200, 1000]:
    albedo = calculate_albedo(ice_thickness, wavelengths, mass_ratio, ice_type="FYI")
    plt.plot(wavelengths, albedo, label=f"{mass_ratio} ng oil/g ice")
plt.ylim([0, 1])
plt.legend()
plt.grid(True)
plt.xlabel("wavelength (nm)")
plt.ylabel("albedo")
plt.savefig("figures/spectral_albedo_realistic_thickness.pdf")
plt.close()

# Alebdo against thickness
plt.figure()
plt.title(f"Spectral albedo of FYI")
for ice_thickness in np.linspace(0.5, 6, 6):
    albedo = calculate_albedo(
        ice_thickness, wavelengths, oil_mass_ratio=0, ice_type="FYI"
    )
    plt.plot(wavelengths, albedo, label=f"pure ice {ice_thickness:.1f}m")
for ice_thickness in np.linspace(0.5, 6, 6):
    albedo = calculate_albedo(
        ice_thickness, wavelengths, oil_mass_ratio=200, ice_type="FYI"
    )
    plt.plot(
        wavelengths, albedo, ls="--", label=f"200ng oil/g ice {ice_thickness:.1f}m"
    )
plt.ylim([0, 1])
plt.legend(fontsize=5)
plt.grid(True)
plt.xlabel("wavelength (nm)")
plt.ylabel("albedo")
plt.savefig("figures/spectral_albedo_against_thickness.pdf")
plt.close()

# Alebdo against thickness at 400nm
plt.figure()
plt.title(f"Spectral albedo of FYI at 400nm")
ice_depths = np.linspace(0.5, 15, 100)
albedo = calculate_albedo(ice_depths, 400, oil_mass_ratio=0, ice_type="FYI")
plt.plot(ice_depths, albedo, label=f"pure ice")
albedo = calculate_albedo(ice_depths, 400, oil_mass_ratio=1000, ice_type="FYI")
plt.plot(ice_depths, albedo, ls="--", label=f"1000ng oil/g ice")
plt.ylim([0, 1])
plt.legend()
plt.grid(True)
plt.xlabel("ice depth (m)")
plt.ylabel("albedo")
plt.savefig("figures/spectral_albedo_against_thickness_at_400nm.pdf")
plt.close()
