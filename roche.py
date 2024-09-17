"""Script to reproduce figure 4 from Redmond Roche 2022

This is a plot of the mass absorption coefficient for different oil types against median droplet radius at different wavelengths
"""

import matplotlib.pyplot as plt
import numpy as np
from oilrad.optics import (
    Romashkino_MAC,
    calculate_ice_oil_absorption_coefficient,
    calculate_ice_oil_extinction_coefficient,
)

if __name__ == "__main__":
    """Reproduce roche 2022 figure 4 but they have given us less data for droplet sizes"""
    plt.figure()
    for wavelength, color in zip(
        [400, 500, 600, 700], ["#8300b5", "#00ff92", "#ffbe00", "#ff0000"]
    ):
        plt.plot(
            [0.05, 0.25, 0.5, 1.5, 2.5, 3.5, 5.0],
            Romashkino_MAC(wavelength, [0.05, 0.25, 0.5, 1.5, 2.5, 3.5, 5.0]),
            color,
            marker="o",
            label=f"{wavelength}nm",
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlim([5e-3, 50])
    plt.ylim([7e-4, 1])
    plt.ylabel("MAC (m^2 g^-1)")
    plt.xlabel("Droplet Radius (microns)")
    plt.title("Romashkino Oil")
    plt.grid(True)
    plt.legend()
    plt.savefig("figures/figure_4_reproduced.pdf")
    plt.close()

    DROPLET_SIZE = 0.5
    ICE_DENSITY = 916
    OIL_MASS_RATIOS = [0, 10, 100, 1000]
    ICE_TYPE = "FYI"

    """Reproduce roche 2022 figure for MAC specturm against wavelength"""
    plt.figure()
    WAVES = np.linspace(350, 800, 10000)
    plt.plot(WAVES, Romashkino_MAC(WAVES, DROPLET_SIZE))
    plt.ylim([1e-3, 1])
    plt.ylabel("MAC (m^2 g^-1)")
    plt.xlabel("Wavelength (nm)")
    plt.yscale("log")
    plt.grid(True)
    plt.title(
        f"Romashkino MAC spectrum for median droplet radius {DROPLET_SIZE} microns"
    )
    plt.savefig("figures/Romashkino_absorption_spectrum.pdf")
    plt.close()

    """Plot ice absoprtion coefficient with oil pollution up to 1000ng/g"""
    WAVES = np.linspace(350, 1500, 1000)
    plt.figure()
    for OIL_MASS_RATIO in OIL_MASS_RATIOS:
        plt.plot(
            WAVES,
            calculate_ice_oil_absorption_coefficient(
                WAVES, OIL_MASS_RATIO, DROPLET_SIZE
            ),
            label=f"{OIL_MASS_RATIO}ng/g",
        )
    plt.ylabel("k (1/m)")
    plt.xlabel("Wavelength (nm)")
    plt.yscale("log")
    plt.grid(True)
    plt.title(f"Romashkino oil in ice median droplet radius {DROPLET_SIZE} mcirons")
    plt.savefig("figures/Roche_oil_absoprtion_coef.pdf")
    plt.close()

    """Plot ice extinction coefficient with oil pollution up to 1000ng/g"""
    plt.figure()
    for OIL_MASS_RATIO in OIL_MASS_RATIOS:
        plt.plot(
            WAVES,
            calculate_ice_oil_extinction_coefficient(
                WAVES, OIL_MASS_RATIO, ICE_TYPE, DROPLET_SIZE
            ),
            label=f"{OIL_MASS_RATIO}ng/g",
        )
    plt.ylabel("mu (1/m)")
    plt.xlabel("Wavelength (nm)")
    plt.yscale("log")
    plt.grid(True)
    plt.title(f"Romashkino oil in ice median droplet radius {DROPLET_SIZE} mcirons")
    plt.savefig("figures/Roche_oil_extinction_coef.pdf")
    plt.close()
