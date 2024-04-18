"""Script to reproduce figure 4 from Redmond Roche 2022

This is a plot of the mass absorption coefficient for different oil types against median droplet radius at different wavelengths
"""

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    Romashkino_droplet_size = [
        "0.05",
        "0.25",
        "0.5",
        "1.5",
        "2.5",
        "3.5",
        "5.0",
    ]

    Petrobaltic_droplet_size = [
        "0.05",
        "0.5",
        "5.0",
    ]

    # dictionary with radius in microns as key and data for MAC against wavelength
    Romashkino = {}
    for droplet_size in Romashkino_droplet_size:
        with open(f"oilrad/data/MassAbsCoe/Romashkino/MAC_{droplet_size}.dat") as file:
            lines = list(file)[1:]
            data = np.loadtxt(lines, delimiter=",")
        Romashkino[droplet_size] = data

    # plt.figure()
    plt.figure()

    for wavelength, color in zip(
        [400, 500, 600, 700], ["#8300b5", "#00ff92", "#ffbe00", "#ff0000"]
    ):
        med_radii = []
        MACs = []
        for droplet_size in Romashkino_droplet_size:
            wavelength_index = np.argmin(
                np.abs(Romashkino[droplet_size][:, 0] - wavelength)
            )
            MAC = Romashkino[droplet_size][wavelength_index, 1]
            MACs.append(MAC)
            med_radii.append(float(droplet_size))
        plt.plot(med_radii, MACs, color, marker="o", label=f"{wavelength}nm")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlim([5e-3, 50])
    plt.ylim([7e-4, 1])
    plt.ylabel("MAC (cm^2 kg^-1)")
    plt.xlabel("Droplet Radius (microns)")
    plt.title("Romashkino Oil")
    plt.grid(True)
    plt.legend()
    plt.savefig("figures/figure_4_reproduced.pdf")
    plt.close()

    plt.figure()
    for SIZE in Romashkino_droplet_size:
        plt.plot(
            Romashkino[SIZE][:, 0],
            Romashkino[SIZE][:, 1],
            label=f"droplet radius {SIZE} microns",
        )
    plt.yscale("log")
    plt.ylim([1e-2, 1])
    plt.ylabel("MAC (cm^2 kg^-1)")
    plt.xlabel("Wavelength (nm)")
    plt.grid(True)
    plt.title(f"Romashkino absorption spectrum")
    plt.legend()
    plt.savefig(f"figures/Romashkino_absorption_spectrum.pdf")
    plt.close()
