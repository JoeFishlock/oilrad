"""Get ice spectral absorption coefficient in 1/m from Warren imaginary refractive
index of ice.

This is the source used by Redmond Roche 2022

This gives comparable if slightly lower extinction coefficients as
Perovich 1990 when using scattering value for white ice interior.
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from oilrad.optics import (
    WARREN_WAVELENGTHS,
    calculate_ice_absorption_coefficient,
    calculate_ice_extinction_coefficient,
)

wavelengths_in_nm = WARREN_WAVELENGTHS * 1e3
absorption_coefficient = calculate_ice_absorption_coefficient(wavelengths_in_nm)
extinction_coefficient_MYI = calculate_ice_extinction_coefficient(
    wavelengths_in_nm, ice_type="MYI"
)
extinction_coefficient_FYI = calculate_ice_extinction_coefficient(
    wavelengths_in_nm, ice_type="FYI"
)
extinction_coefficient_MELT = calculate_ice_extinction_coefficient(
    wavelengths_in_nm, ice_type="MELT"
)
# plot ice absorption data and show it absorps longwave strongly
plt.figure()
ax = plt.gca()
plt.title("Ice absorption coefficient")
plt.plot(wavelengths_in_nm, absorption_coefficient, marker="o")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("wavelength (nm)")
plt.ylabel("absorption coefficient (1/m)")
plt.xlim([300, 100000])
plt.ylim([1e-4, 1e8])
plt.grid(True)
plt.savefig("figures/spectral_ice_absorption.pdf")
plt.close()

plt.figure()
ax = plt.gca()
plt.plot(
    wavelengths_in_nm,
    extinction_coefficient_MELT,
    marker="o",
    label="pure MELT ice extinction coefficient",
)
plt.plot(
    wavelengths_in_nm,
    extinction_coefficient_FYI,
    marker="o",
    label="pure FYI ice extinction coefficient",
)
plt.plot(
    wavelengths_in_nm,
    extinction_coefficient_MYI,
    marker="o",
    label="pure MYI ice extinction coefficient",
)
plt.plot(
    wavelengths_in_nm,
    absorption_coefficient,
    marker="o",
    label="absorption coefficient",
)
plt.xscale("log")
plt.yscale("log")
plt.xlim([350, 1000])
plt.ylim([1e-4, 1e2])
plt.ylabel("(1/m)")
plt.xlabel("wavelength (nm)")
plt.grid(True)
ax.xaxis.set_minor_formatter(mticker.StrMethodFormatter("{x:.0f}"))
plt.legend()
plt.savefig("figures/spectral_ice_extinction.pdf")
plt.close()
